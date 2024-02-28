import os
import numpy as np

import torch
import torch.nn as nn
from torch.nn import functional as F
from .segment import SegmentModel, SegmentLabel
from .cwct import CWCT
from .color import rgb2lab, lab2rgb

from typing import List

import todos
import pdb

def vstnet_split(x) -> List[torch.Tensor]:
    n = x.size(1)//2
    x1 = x[:, :n, :, :].contiguous()
    x2 = x[:, n:, :, :].contiguous()
    return x1, x2


def vstnet_merge(x1, x2):
    return torch.cat((x1, x2), dim=1)


def vstnet_squeeze(x, size:int=2):
    B, C, H, W = x.size()
    H = H // size
    W = W // size
    x = x.reshape(B, C, H, size, W, size).permute(0, 3, 5, 1, 2, 4)
    return x.reshape(B, C*size*size, H, W)


def vstnet_unsqueeze(x, size:int =2):
    B, C, H, W = x.size()
    C = C // (size * size)
    x = x.reshape(B, size, size, C, H, W).permute(0, 3, 4, 1, 5, 2)
    return x.reshape(B, C, H * size, W * size)


class InjectivePad(nn.Module):
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


class ResidualBlock1(nn.Module):
    def __init__(self, channel, mult=4, kernel=3):
        super().__init__()
        stride = 1
        in_ch = channel

        self.conv = nn.Sequential(
            nn.ReflectionPad2d((kernel - 1) // 2),
            nn.Conv2d(in_ch, channel//mult, kernel_size=kernel, stride=stride, padding=0, bias=True),
            nn.ReLU(),
            nn.ReflectionPad2d((kernel - 1) // 2),
            nn.Conv2d(channel // mult, channel // mult, kernel_size=kernel, padding=0, bias=True),
            nn.ReLU(),
            nn.ReflectionPad2d((kernel - 1) // 2),
            nn.Conv2d(channel // mult, channel, kernel_size=kernel, padding=0, bias=True)
        )

    def forward(self, x: List[torch.Tensor]) -> List[torch.Tensor]:
        x1 = x[0]
        x2 = x[1]
        Fx2 = self.conv(x2)
        return (x2, Fx2 + x1)

    def inverse(self, x: List[torch.Tensor]) -> List[torch.Tensor]:
        x2, y1 = x[0], x[1]
        Fx2 = - self.conv(x2)
        x1 = Fx2 + y1
        return (x1, x2)

class ResidualBlock2(nn.Module):
    def __init__(self, channel, mult=4, kernel=3):
        super().__init__()
        stride = 2
        in_ch = channel // 4

        self.conv = nn.Sequential(
            nn.ReflectionPad2d((kernel - 1) // 2),
            nn.Conv2d(in_ch, channel//mult, kernel_size=kernel, stride=stride, padding=0, bias=True),
            nn.ReLU(),
            nn.ReflectionPad2d((kernel - 1) // 2),
            nn.Conv2d(channel // mult, channel // mult, kernel_size=kernel, padding=0, bias=True),
            nn.ReLU(),
            nn.ReflectionPad2d((kernel - 1) // 2),
            nn.Conv2d(channel // mult, channel, kernel_size=kernel, padding=0, bias=True)
        )

    def forward(self, x: List[torch.Tensor]) -> List[torch.Tensor]:
        x1 = x[0]
        x2 = x[1]
        Fx2 = self.conv(x2)
        x1 = vstnet_squeeze(x1) # [1, 16, 676, 1200] ==> [1, 64, 338, 600]
        x2 = vstnet_squeeze(x2) # [1, 16, 676, 1200] ==> [1, 64, 338, 600]
        return (x2, Fx2 + x1)

    def inverse(self, x: List[torch.Tensor]) -> List[torch.Tensor]:
        x2, y1 = x[0], x[1]
        x2 = vstnet_unsqueeze(x2)
        Fx2 = - self.conv(x2)
        x1 = Fx2 + y1
        x1 = vstnet_unsqueeze(x1)
        return (x1, x2)


class ChannelReduction(nn.Module):
    def __init__(self, in_ch, out_ch, sp_steps=2, n_blocks=2, kernel=3):
        super().__init__()
        pad = out_ch * (4 ** sp_steps) - in_ch
        self.inj_pad = InjectivePad(pad)
        self.sp_steps = sp_steps

        self.block_list = nn.ModuleList()
        for i in range(n_blocks):
            self.block_list.append(ResidualBlock1(out_ch * 4 ** sp_steps, mult=4, kernel=kernel))

    def forward(self, x):
        x = list(vstnet_split(x))
        x[0] = self.inj_pad.forward(x[0])
        x[1] = self.inj_pad.forward(x[1])

        # support torch.jit.script
        # for block in self.block_list:
        #     x = block.forward(x)
        for i, block in enumerate(self.block_list):
            x = block(x)

        x = vstnet_merge(x[0], x[1])

        # spread
        for _ in range(self.sp_steps):
            B, C, H, W = x.size()
            C = C // 4 # 4 === 2*2
            x = x.reshape(B, 2, 2, C, H, W).permute(0, 3, 4, 1, 5, 2)
            x = x.reshape(B, C, H * 2, W * 2)

        return x

    def inverse(self, x):
        for _ in range(self.sp_steps):
            B, C, H, W = x.size()
            H = H // 2
            W = W // 2
            x = x.reshape(B, C, H, 2, W, 2).permute(0, 3, 5, 1, 2, 4)
            x = x.reshape(B, C * 4, H, W)

        x = vstnet_split(x)
        # support torch.jit.script
        # for block in self.block_list[::-1]:
        #     x = block.inverse(x)
        n = len(self.block_list) # 2
        for i in range(n):
            for j, block in enumerate(self.block_list):
                if j == n - i - 1:
                    x = block(x)
        x = list(x)
        x[0] = self.inj_pad.inverse(x[0])
        x[1] = self.inj_pad.inverse(x[1])

        x = vstnet_merge(x[0], x[1])
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

        pad = 2 * nChannels[0] - in_channel
        self.inj_pad = InjectivePad(pad)

        self.stack = self.block_stack(nChannels, nBlocks, nStrides, mult=mult, kernel=kernel)
        self.channel_reduction = ChannelReduction(nChannels[-1], hidden_dim, sp_steps=sp_steps, kernel=kernel)

        self.load_weights(model_path=model_path)

        self.segment_model = SegmentModel()
        self.segment_label = SegmentLabel()
        self.cwct_model = CWCT()


    def load_weights(self, model_path="models/image_photo_style.pth"):
        cdir = os.path.dirname(__file__)
        checkpoint = model_path if cdir == "" else cdir + "/" + model_path
        sd = torch.load(checkpoint)
        if 'state_dict' in sd.keys():
            sd = sd['state_dict']
        self.load_state_dict(sd)


    def block_stack(self, nChannels, nBlocks, nStrides, mult, kernel=3):
        block_list = nn.ModuleList()
        strides = []
        channels = []
        for channel, depth, stride in zip(nChannels, nBlocks, nStrides):
            strides = strides + ([stride] + [1]*(depth-1))
            channels = channels + ([channel]*depth)
        for channel, stride in zip(channels, strides):
            if stride == 1:
                block_list.append(ResidualBlock1(channel, mult=mult, kernel=kernel))
            else:
                block_list.append(ResidualBlock2(channel, mult=mult, kernel=kernel))

        return block_list

    def resize_pad_tensor(self, x):
        # Need Resize ?
        B, C, H, W = x.size()
        s = min(min(self.MAX_H / H, self.MAX_W / W), 1.0)
        SH, SW = int(s * H), int(s * W)
        resize_x = F.interpolate(x, size=(SH, SW), mode="bilinear", align_corners=False)

        # Need Pad ?
        r_pad = (self.MAX_TIMES - (SW % self.MAX_TIMES)) % self.MAX_TIMES
        b_pad = (self.MAX_TIMES - (SH % self.MAX_TIMES)) % self.MAX_TIMES
        resize_pad_x = F.pad(resize_x, (0, r_pad, 0, b_pad), mode="replicate")
        return resize_pad_x


    def forward(self, content, style):
        B, C, H, W = content.size()
        content_lab = rgb2lab(content)

        content = self.resize_pad_tensor(content)
        style = self.resize_pad_tensor(style)

        # Encode features
        z_c = self.encode(content)
        z_s = self.encode(style)

        # Segment and simple
        content_seg = self.segment_model(content)
        content_seg = self.segment_label(content_seg) # remove small holes
        style_seg = self.segment_model(style)
        style_reg = self.segment_label(style_seg) # remove small holes

        # Refine content segment via style segment guide
        content_seg = self.segment_label.cross_remapping(content_seg, style_seg)

        z_cs = self.cwct_model(z_c, z_s, content_seg, style_seg)

        output = self.decode(z_cs)
        output = F.interpolate(output, size=(H, W), mode="bilinear", align_corners=False)

        output_lab = rgb2lab(output)
        blend_lab = torch.cat((content_lab[:, 0:1, :, :], output_lab[:, 1:3, :, :]), dim=1)
        output = lab2rgb(blend_lab)

        return output


    def encode(self, x):
        # tensor [x] size: [1, 3, 676, 1200], min: 0.0, max: 1.0, mean: 0.331319
        x = self.inj_pad.forward(x)

        x = vstnet_split(x)
        for block in self.stack:
            x = block.forward(x)

        x = vstnet_merge(x[0], x[1])

        x = self.channel_reduction.forward(x)

        # tensor [x] size: [1, 32, 676, 1200], min: -1.037655, max: 1.071546, mean: -0.001224
        return x

    def decode(self, x):
        # tensor [x] size: [1, 32, 676, 1200], min: -1.211427, max: 1.173751, mean: 0.000158

        x = self.channel_reduction.inverse(x)

        x = vstnet_split(x)
        # ugly code for torch.jit.script not support inverse !!!
        n = len(self.stack) # 30
        for i in range(n):
            #x = self.stack[-1-i].inverse(x)
            for j, block in enumerate(self.stack):
                if j == n - i - 1:
                    x = block.inverse(x)
        x = vstnet_merge(x[0], x[1])

        x = self.inj_pad.inverse(x)

        # tensor [x] size: [1, 3, 676, 1200], min: -0.478305, max: 1.713724, mean: 0.482254
        return x


def create_photo_style_model():
    model = RevResNet(hidden_dim=16, sp_steps=2, model_path="models/image_photo_style.pth")
    return model


def create_artist_style_model():
    model = RevResNet(hidden_dim=64, sp_steps=1, model_path="models/image_artist_style.pth")
    return model
