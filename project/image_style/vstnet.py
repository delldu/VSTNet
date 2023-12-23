import os
import torch
import torch.nn as nn
import numpy as np
from torch.nn import functional as F
from .segment import SegmentModel, SegmentLabel
from .cwct import CWCT

# from typing import Tuple
import todos
import pdb

def split(x):
    n = int(x.size()[1]/2)
    x1 = x[:, :n, :, :].contiguous()
    x2 = x[:, n:, :, :].contiguous()
    return x1, x2


def merge(x1, x2):
    return torch.cat((x1, x2), 1)


class injective_pad(nn.Module):
    def __init__(self, pad_size):
        super().__init__()
        self.pad_size = pad_size
        self.pad = nn.ZeroPad2d((0, 0, 0, pad_size))

    def forward(self, x):
        x = x.permute(0, 2, 1, 3)
        x = self.pad(x)
        return x.permute(0, 2, 1, 3)

    def inverse(self, x):
        return x[:, :x.size(1) - self.pad_size, :, :]


def squeeze(x, size=2):
    bs, d, new_h, new_w = x.shape[0], x.shape[1], x.shape[2]//size, x.shape[3]//size
    x = x.reshape(bs, d, new_h, size, new_w, size).permute(0, 3, 5, 1, 2, 4)
    return x.reshape(bs, d*(size**2), new_h, new_w)


def unsqueeze(x, size=2):
    bs, new_d, h, w = x.shape[0], x.shape[1]//(size**2), x.shape[2], x.shape[3]
    x = x.reshape(bs, size, size, new_d, h, w).permute(0, 3, 4, 1, 5, 2)
    return x.reshape(bs, new_d, h * size, w * size)


class InvConv2d(nn.Module):
    def __init__(self, channel):
        """ Invertible MLP """
        super().__init__()

        weight = torch.randn(channel, channel)
        bias = torch.randn(1, channel, 1, 1)
        q, _ = torch.linalg.qr(weight)
        weight = q.unsqueeze(2).unsqueeze(3)
        self.weight = nn.Parameter(weight)
        self.bias = nn.Parameter(bias)

    def forward(self, x):
        _, _, height, width = x.shape
        out = F.conv2d(x, self.weight)
        return out + self.bias

    def inverse(self, y):
        y = y - self.bias
        return F.conv2d(
            y, self.weight.squeeze().inverse().unsqueeze(2).unsqueeze(3)
        )


class residual_block(nn.Module):
    def __init__(self, channel, stride=1, mult=4, kernel=3):
        super().__init__()
        self.stride = stride

        pad = (kernel - 1) // 2
        if stride == 1:
            in_ch = channel
        else:
            in_ch = channel // 4

        self.conv = nn.Sequential(
            nn.ReflectionPad2d(pad),
            nn.Conv2d(in_ch, channel//mult, kernel_size=kernel, stride=stride, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(pad),
            nn.Conv2d(channel // mult, channel // mult, kernel_size=kernel, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(pad),
            nn.Conv2d(channel // mult, channel, kernel_size=kernel, padding=0, bias=True)
        )
        self.init_layers()

    def init_layers(self):
        for m in self.conv:
            if isinstance(m, nn.Conv2d):
                m.bias.data.zero_()

    def forward(self, x):
        x1 = x[0]
        x2 = x[1]
        Fx2 = self.conv(x2)
        if self.stride == 2:
            x1 = squeeze(x1)
            x2 = squeeze(x2)
        y1 = Fx2 + x1
        return (x2, y1)

    def inverse(self, x):
        x2, y1 = x[0], x[1]
        if self.stride == 2:
            x2 = unsqueeze(x2)
        Fx2 = - self.conv(x2)
        x1 = Fx2 + y1
        if self.stride == 2:
            x1 = unsqueeze(x1)

        x = (x1, x2)
        return x


class channel_reduction(nn.Module):
    def __init__(self, in_ch, out_ch, sp_steps=2, n_blocks=2, kernel=3):
        super().__init__()
        self.pad = out_ch * 4 ** sp_steps - in_ch
        self.inj_pad = injective_pad(self.pad)
        self.sp_steps = sp_steps
        self.n_blocks = n_blocks

        self.block_list = nn.ModuleList()
        for i in range(n_blocks):
            self.block_list.append(residual_block(out_ch * 4 ** sp_steps, stride=1, mult=4, kernel=kernel))

    def forward(self, x):
        x = list(split(x))
        x[0] = self.inj_pad.forward(x[0])
        x[1] = self.inj_pad.forward(x[1])

        for block in self.block_list:
            x = block.forward(x)
        x = merge(x[0], x[1])

        # spread
        for _ in range(self.sp_steps):
            bs, new_d, h, w = x.shape[0], x.shape[1]//2**2, x.shape[2], x.shape[3]
            x = x.reshape(bs, 2, 2, new_d, h, w).permute(0, 3, 4, 1, 5, 2)
            x = x.reshape(bs, new_d, h * 2, w * 2)

        return x

    def inverse(self, x):
        for _ in range(self.sp_steps):
            bs, d, new_h, new_w = x.shape[0], x.shape[1], x.shape[2]//2, x.shape[3]//2
            x = x.reshape(bs, d, new_h, 2, new_w, 2).permute(0, 3, 5, 1, 2, 4)
            x = x.reshape(bs, d * 2**2, new_h, new_w)

        x = split(x)
        for block in self.block_list[::-1]:
            x = block.inverse(x)

        x = list(x)
        x[0] = self.inj_pad.inverse(x[0])
        x[1] = self.inj_pad.inverse(x[1])

        x = merge(x[0], x[1])
        return x


class RevResNet(nn.Module):
    '''Reversible Residual Network'''
    def __init__(self,
            nBlocks=[10, 10, 10],
            nStrides=[1, 2, 2], 
            nChannels=[16, 64, 256], 
            in_channel=3, 
            mult=4, 
            hidden_dim=16, 
            sp_steps=2, 
            kernel=3,
            model_path="models/image_photo_style.pth",
        ):
        super().__init__()
        self.MAX_H = 1280
        self.MAX_W = 1280
        self.MAX_TIMES = 4
        # GPU -- ?

        self.nBlocks = nBlocks
        self.pad = 2 * nChannels[0] - in_channel
        self.inj_pad = injective_pad(self.pad)
        self.in_ch = nChannels[0]
        self.down_scale = np.prod(np.array(nStrides))

        self.stack = self.block_stack(residual_block, nChannels, nBlocks, nStrides, mult=mult, kernel=kernel)

        self.channel_reduction = channel_reduction(nChannels[-1], hidden_dim, sp_steps=sp_steps, kernel=kernel)

        self.load_weights(model_path=model_path)

        self.segment_model = SegmentModel()
        self.segment_label = SegmentLabel()
        self.wct_model = CWCT()



    def load_weights(self, model_path="models/image_photo_style.pth"):
        cdir = os.path.dirname(__file__)
        checkpoint = model_path if cdir == "" else cdir + "/" + model_path
        sd = torch.load(checkpoint)
        if 'state_dict' in sd.keys():
            sd = sd['state_dict']
        self.load_state_dict(sd)


    def block_stack(self, _block, nChannels, nBlocks, nStrides, mult, kernel=3):
        block_list = nn.ModuleList()
        strides = []
        channels = []
        for channel, depth, stride in zip(nChannels, nBlocks, nStrides):
            strides = strides + ([stride] + [1]*(depth-1))
            channels = channels + ([channel]*depth)
        for channel, stride in zip(channels, strides):
            block_list.append(_block(channel, stride, mult=mult, kernel=kernel))
        return block_list

    def resize_pad_tensor(self, x):
        # Need Resize ?
        B, C, H, W = x.size()
        s = min(min(self.MAX_H / H, self.MAX_W / W), 1.0)
        SH, SW = int(s * H), int(s * W)
        resize_x = F.interpolate(x.float(), size=(SH, SW), mode="bilinear", align_corners=False).to(x.dtype)

        # Need Pad ?
        r_pad = (self.MAX_TIMES - (SW % self.MAX_TIMES)) % self.MAX_TIMES
        b_pad = (self.MAX_TIMES - (SH % self.MAX_TIMES)) % self.MAX_TIMES
        resize_pad_x = F.pad(resize_x, (0, r_pad, 0, b_pad), mode="replicate")
        return resize_pad_x


    def forward(self, content, style):
        B, C, H, W = content.size()

        content = self.resize_pad_tensor(content)
        style = self.resize_pad_tensor(style)

        # todos.debug.output_var("content", content)
        # todos.debug.output_var("style", style)

        # Encode feature
        content_feat = self.encode(content)
        style_feat = self.encode(style)

        # Segment and simple
        content_seg = self.segment_model(content)
        content_seg = self.segment_label(content_seg)
        style_seg = self.segment_model(style)
        style_reg = self.segment_label(style_seg)

        # Refine content segment via style segment
        content_seg = self.segment_label.cross_remapping(content_seg, style_seg)

        z_cs = self.wct_model(content_feat, style_feat, content_seg, style_seg)

        output = self.decode(z_cs)
        return F.interpolate(output, size=(H, W), mode="bilinear", align_corners=False)


    def encode(self, x):
        x = self.inj_pad.forward(x)

        x = split(x)
        for block in self.stack:
            x = block.forward(x)

        x = merge(x[0], x[1])

        x = self.channel_reduction.forward(x)

        return x

    def decode(self, x):
        x = self.channel_reduction.inverse(x)

        x = split(x)
        for i in range(len(self.stack)):
            x = self.stack[-1-i].inverse(x)
        x = merge(x[0], x[1])

        x = self.inj_pad.inverse(x)

        return x


def create_photo_style_model():
    model = RevResNet(hidden_dim=16, sp_steps=2, model_path="models/image_photo_style.pth")
    return model


def create_artist_style_model():
    model = RevResNet(hidden_dim=64, sp_steps=1, model_path="models/image_artist_style.pth")
    return model
