import os

import torch
import torch.nn as nn
from torch.nn import functional as F
from .segment import SegmentModel
from .cwct import CWCT
from .color import rgb2lab, lab2rgb

from typing import Tuple

import todos
import pdb


def vstnet_split(x) -> Tuple[torch.Tensor, torch.Tensor]:
    n = x.size(1) // 2
    x1 = x[:, :n, :, :].contiguous()
    x2 = x[:, n:, :, :].contiguous()
    return (x1, x2)


def vstnet_merge(x1, x2):
    return torch.cat((x1, x2), dim=1)


def vstnet_pixel_shuffle(x, size: int = 2):
    B, C, H, W = x.size()
    H = H // size
    W = W // size
    x = x.reshape(B, C, H, size, W, size).permute(0, 3, 5, 1, 2, 4)
    return x.reshape(B, C * size * size, H, W)


def vstnet_pixel_unshuffle(x, size: int = 2):
    B, C, H, W = x.size()
    C = C // (size * size)
    x = x.reshape(B, size, size, C, H, W).permute(0, 3, 4, 1, 5, 2)
    return x.reshape(B, C, H * size, W * size)


class InjectivePad(nn.Module):
    def __init__(self, pad_size):
        super().__init__()
        self.pad_size = pad_size  # 0 or 29
        self.pad = nn.ZeroPad2d((0, 0, 0, pad_size))

    def forward(self, x):
        # x.size() -- [1, 3, 576, 1024]
        x = x.permute(0, 2, 1, 3)
        x = self.pad(x)
        return x.permute(0, 2, 1, 3)

    def inverse(self, x):
        return x[:, : x.size(1) - self.pad_size, :, :]


class ResidualBlock1(nn.Module):
    def __init__(self, channel):
        super().__init__()
        stride = 1
        in_ch = channel

        self.conv = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_ch, channel // 4, kernel_size=3, stride=stride, padding=0, bias=True),
            nn.ReLU(),
            nn.ReflectionPad2d(1),
            nn.Conv2d(channel // 4, channel // 4, kernel_size=3, padding=0, bias=True),
            nn.ReLU(),
            nn.ReflectionPad2d(1),
            nn.Conv2d(channel // 4, channel, kernel_size=3, padding=0, bias=True),
        )

    def forward(self, x1, x2) -> Tuple[torch.Tensor, torch.Tensor]:
        fx = self.conv(x2)
        return (x2, fx + x1)

    def inverse(self, y1, y2) -> Tuple[torch.Tensor, torch.Tensor]:
        fx = self.conv(y1)
        return (y2 - fx, y1)


class ResidualBlock2(nn.Module):
    def __init__(self, channel):
        super().__init__()
        stride = 2
        in_ch = channel // 4

        self.conv = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_ch, channel // 4, kernel_size=3, stride=stride, padding=0, bias=True),
            nn.ReLU(),
            nn.ReflectionPad2d(1),
            nn.Conv2d(channel // 4, channel // 4, kernel_size=3, padding=0, bias=True),
            nn.ReLU(),
            nn.ReflectionPad2d(1),
            nn.Conv2d(channel // 4, channel, kernel_size=3, padding=0, bias=True),
        )

    def forward(self, x1, x2) -> Tuple[torch.Tensor, torch.Tensor]:
        fx = self.conv(x2)
        x1 = vstnet_pixel_shuffle(x1)  # [1, 16, 676, 1200] ==> [1, 64, 338, 600]
        x2 = vstnet_pixel_shuffle(x2)  # [1, 16, 676, 1200] ==> [1, 64, 338, 600]
        return (x2, fx + x1)

    def inverse(self, y1, y2) -> Tuple[torch.Tensor, torch.Tensor]:
        y1 = vstnet_pixel_unshuffle(y1)
        fx = self.conv(y1)
        x1 = y2 - fx
        x1 = vstnet_pixel_unshuffle(x1)
        return (x1, y1)


class ChannelReduction(nn.Module):
    def __init__(self, in_ch=256, out_ch=16, sp_steps=2, n_blocks=2):
        super().__init__()
        pad = out_ch * (4**sp_steps) - in_ch  # ==> pad === 0
        self.inj_pad = InjectivePad(pad)
        self.sp_steps = sp_steps

        self.block_list = nn.ModuleList()
        for i in range(n_blocks):
            self.block_list.append(ResidualBlock1(out_ch * 4**sp_steps))

    def forward(self, x):
        # tensor [x] size: [1, 512, 144, 256], min: -0.669773, max: 0.737936, mean: -0.000687

        x1, x2 = vstnet_split(x)
        x1 = self.inj_pad.forward(x1)
        x2 = self.inj_pad.forward(x2)

        # support torch.jit.script
        # for block in self.block_list:
        #     x = block.forward(x)
        for i, block in enumerate(self.block_list):
            (x1, x2) = block(x1, x2)

        x = vstnet_merge(x1, x2)

        # spread
        for _ in range(self.sp_steps):
            x = vstnet_pixel_unshuffle(x)

        # tensor [x] size: [1, 32, 576, 1024], min: -0.919658, max: 1.018641, mean: -0.001065
        return x

    def inverse(self, x):
        for _ in range(self.sp_steps):
            x = vstnet_pixel_shuffle(x, size=2)

        x1, x2 = vstnet_split(x)
        # support torch.jit.script
        # for block in self.block_list[::-1]:
        #     x = block.inverse(x)
        n = len(self.block_list)  # 2
        for i in range(n):
            for j, block in enumerate(self.block_list):
                if j == n - i - 1:
                    x1, x2 = block(x1, x2)
        # x = list(x)
        x1 = self.inj_pad.inverse(x1)
        x2 = self.inj_pad.inverse(x2)

        x = vstnet_merge(x1, x2)
        return x


class VSTNetModel(nn.Module):
    """Versatile Style Transfer Network"""

    def __init__(
        self,
        hidden_dim=16,
        sp_steps=2,
        model_path="models/image_photo_style.pth",
    ):
        super().__init__()
        self.MAX_H = 1536
        self.MAX_W = 1536
        self.MAX_TIMES = 4
        # GPU -- 1024x1024, 5G, 800 ms
        # GPU -- 1536x1536, 8.6G, 2650 ms

        self.encoder = VSTEncoder(hidden_dim=hidden_dim, sp_steps=sp_steps, model_path=model_path)
        self.decoder = VSTDecoder(hidden_dim=hidden_dim, sp_steps=sp_steps, model_path=model_path)

        self.segment_model = SegmentModel()
        # self.segment_label = SegmentLabel()
        self.cwct_model = CWCT()

    def pad_tensor(self, x):
        # Need Resize ?
        B, C, H, W = x.size()

        # Need Pad ?
        r_pad = (self.MAX_TIMES - (W % self.MAX_TIMES)) % self.MAX_TIMES
        b_pad = (self.MAX_TIMES - (H % self.MAX_TIMES)) % self.MAX_TIMES
        return F.pad(x, (0, r_pad, 0, b_pad), mode="replicate")

    def forward(self, c_image, s_image):
        B, C, H, W = c_image.size()
        content_lab = rgb2lab(c_image)  # size() -- [1, 3, 675, 1200]
        # tensor [c_image] size: [1, 3, 675, 1200], min: 0.0, max: 1.0, mean: 0.331631

        s_image = self.pad_tensor(s_image)
        c_image = self.pad_tensor(c_image)  # size() -- [1, 3, 576, 1024]

        # Encode features
        s_feat = self.encoder.forward(s_image)  # self.encode(s_image)
        c_feat = self.encoder.forward(c_image)  # size() -- [1, 32, 576, 1024]

        # Segment and simple
        c_mask = self.segment_model(c_image).to(torch.int64)  # size() -- [1, 1, 576, 1024]
        c_mask = self.segment_model.remove_small_holes(c_mask)  # remove small holes
        s_mask = self.segment_model(s_image).to(torch.int64)
        s_mask = self.segment_model.remove_small_holes(s_mask)  # remove small holes

        z_cs = self.cwct_model(c_feat, s_feat, c_mask, s_mask)  # size() -- [1, 32, 576, 1024]

        output = self.decoder.forward(z_cs)  # size() -- [1, 3, 576, 1024]
        output = F.interpolate(output, size=(H, W), mode="bilinear", align_corners=False)

        output_lab = rgb2lab(output)

        # blend_ab = 0.1*content_lab[:, 1:3, :, :] + 0.9*output_lab[:, 1:3, :, :]
        blend_ab = output_lab[:, 1:3, :, :]

        blend_lab = torch.cat((content_lab[:, 0:1, :, :], blend_ab), dim=1)
        output = lab2rgb(blend_lab)

        return output


class VSTAE(nn.Module):
    """VSTAE ---- Versatile Style Transfer Auto Encoder"""

    def __init__(self, hidden_dim=16, sp_steps=2, model_path="models/image_photo_style.pth"):
        super().__init__()
        nBlocks = [10, 10, 10]
        nStrides = [1, 2, 2]
        nChannels = [16, 64, 256]
        in_channel = 3

        pad = 2 * nChannels[0] - in_channel  # 29
        self.inj_pad = InjectivePad(pad)
        self.stack = self.block_stack(nChannels, nBlocks, nStrides)
        self.channel_reduction = ChannelReduction(nChannels[-1], hidden_dim, sp_steps=sp_steps)

        self.load_weights(model_path=model_path)

    def block_stack(self, nChannels, nBlocks, nStrides):
        block_list = nn.ModuleList()
        strides = []
        channels = []
        for channel, depth, stride in zip(nChannels, nBlocks, nStrides):
            strides = strides + ([stride] + [1] * (depth - 1))
            channels = channels + ([channel] * depth)

        for channel, stride in zip(channels, strides):
            if stride == 1:
                block_list.append(ResidualBlock1(channel))
            else:
                block_list.append(ResidualBlock2(channel))

        return block_list

    def load_weights(self, model_path="models/image_photo_style.pth"):
        cdir = os.path.dirname(__file__)
        checkpoint = model_path if cdir == "" else cdir + "/" + model_path
        sd = torch.load(checkpoint)
        if "state_dict" in sd.keys():
            sd = sd["state_dict"]
        self.load_state_dict(sd)
        print(f"Loading {model_path} OK.")

    def forward(self, x):
        # useless, just place holder function
        return x


class VSTEncoder(VSTAE):
    """Versatile Style Transfer Encoder"""

    def __init__(self, hidden_dim=16, sp_steps=2, model_path="models/image_photo_style.pth"):
        super().__init__(hidden_dim=hidden_dim, sp_steps=sp_steps, model_path=model_path)

    def forward(self, x):
        # tensor [x] size: [1, 3, 676, 1200], min: 0.0, max: 1.0, mean: 0.331319
        x = self.inj_pad.forward(x)

        x1, x2 = vstnet_split(x)
        for block in self.stack:
            x1, x2 = block(x1, x2)
        x = vstnet_merge(x1, x2)

        x = self.channel_reduction.forward(x)

        # tensor [x] size: [1, 32, 676, 1200], min: -1.037655, max: 1.071546, mean: -0.001224
        return x


class VSTDecoder(VSTAE):
    """Versatile Style Transfer Decoder"""

    def __init__(self, hidden_dim=16, sp_steps=2, model_path="models/image_photo_style.pth"):
        super().__init__(hidden_dim=hidden_dim, sp_steps=sp_steps, model_path=model_path)

        self.reverse_stack = nn.ModuleList()
        for n, m in self.stack.named_children():
            self.reverse_stack.insert(0, m)

    def forward(self, x):
        # tensor [x] size: [1, 32, 676, 1200], min: -1.211427, max: 1.173751, mean: 0.000158

        x = self.channel_reduction.inverse(x)

        x1, x2 = vstnet_split(x)
        # # ugly code for torch.jit.script not support inverse !!!
        # n = len(self.stack) # 30
        # for i in range(n):
        #     #x = self.stack[-1-i].inverse(x)
        #     for j, block in enumerate(self.stack):
        #         if j == n - i - 1:
        #             x1, x2 = block.inverse(x1, x2)
        for block in self.reverse_stack:
            x1, x2 = block.inverse(x1, x2)

        x = vstnet_merge(x1, x2)

        x = self.inj_pad.inverse(x)

        # tensor [x] size: [1, 3, 676, 1200], min: -0.478305, max: 1.713724, mean: 0.482254
        return x.clamp(0.0, 1.0)


def create_photo_style_model():
    model = VSTNetModel(hidden_dim=16, sp_steps=2, model_path="models/image_photo_style.pth")
    return model


def create_artist_style_model():
    model = VSTNetModel(hidden_dim=64, sp_steps=1, model_path="models/image_artist_style.pth")
    return model
