"""Create model."""  # coding=utf-8
#
# /************************************************************************************
# ***
# ***    Copyright Dell 2022, All Rights Reserved.
# ***
# ***    File Author: Dell, 2022年 09月 27日 星期二 02:00:52 CST
# ***
# ************************************************************************************/
#

import pdb  # For debug
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms.functional import normalize
import numpy as np
from typing import List
from functools import partial



class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(0.0)

    def forward(self, x, H: int, W: int):
        x = self.fc1(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=True)
        self.kv = nn.Linear(dim, dim * 2, bias=True)
        self.attn_drop = nn.Dropout(0.0)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(0.0)

        self.sr_ratio = sr_ratio # maybe 8, 4, 2, 1
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)
        else:  # Support torch.jit.script
            self.sr = nn.Identity()
            self.norm = nn.Identity()

    def forward(self, x, H: int, W: int):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class Block(nn.Module):
    def __init__(self, dim, num_heads,
        mlp_ratio=4.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        sr_ratio=1,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, sr_ratio=sr_ratio)

        self.drop_path = nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer)

    def forward(self, x, H: int, W: int):
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))

        return x


class OverlapPatchEmbed(nn.Module):
    """Image to Patch Embedding"""

    def __init__(self, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        patch_size = (patch_size, patch_size)

        self.patch_size = patch_size # eg: (7, 7), (3, 3)
        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=stride,
            padding=(patch_size[0] // 2, patch_size[1] // 2),
        )
        self.norm = nn.LayerNorm(embed_dim)


    def forward(self, x) -> List[torch.Tensor]:
        x = self.proj(x)
        proj_out = x
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)

        return x, proj_out


class VisionTransformer(nn.Module):
    def __init__(self,
        patch_size=16,
        in_chans=3,
        num_classes=1000,
        embed_dims=[64, 128, 256, 512],
        num_heads=[1, 2, 4, 8],
        mlp_ratios=[4, 4, 4, 4],
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
        depths=[3, 4, 6, 3],
        sr_ratios=[8, 4, 2, 1],
        embedding_dim=768,
    ):
        super().__init__()

        # self = mit_b2()
        # patch_size = 4
        # in_chans = 3
        # num_classes = 1000
        # embed_dims = [64, 128, 320, 512]
        # num_heads = [1, 2, 5, 8]
        # mlp_ratios = [4, 4, 4, 4]
        # drop_path_rate = 0.1
        # norm_layer = functools.partial(<class 'torch.nn.modules.normalization.LayerNorm'>, eps=1e-06)
        # depths = [3, 4, 6, 3]
        # sr_ratios = [8, 4, 2, 1]

        self.num_classes = num_classes
        self.depths = depths
        self.embedding_dim = embedding_dim

        # patch_embed
        self.patch_embed1 = OverlapPatchEmbed(patch_size=7, stride=4, in_chans=in_chans, embed_dim=embed_dims[0])
        self.patch_embed2 = OverlapPatchEmbed(patch_size=3, stride=2, in_chans=embed_dims[0], embed_dim=embed_dims[1])
        self.patch_embed3 = OverlapPatchEmbed(patch_size=3, stride=2, in_chans=embed_dims[1], embed_dim=embed_dims[2])
        self.patch_embed4 = OverlapPatchEmbed(patch_size=3, stride=2, in_chans=embed_dims[2], embed_dim=embed_dims[3])

        # transformer encoder
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0
        self.block1 = nn.ModuleList(
            [
                Block(
                    dim=embed_dims[0],
                    num_heads=num_heads[0],
                    mlp_ratio=mlp_ratios[0],
                    drop_path=dpr[cur + i],
                    norm_layer=norm_layer,
                    sr_ratio=sr_ratios[0],
                )
                for i in range(depths[0])
            ]
        )
        self.norm1 = norm_layer(embed_dims[0])

        cur += depths[0]
        self.block2 = nn.ModuleList(
            [
                Block(
                    dim=embed_dims[1],
                    num_heads=num_heads[1],
                    mlp_ratio=mlp_ratios[1],
                    drop_path=dpr[cur + i],
                    norm_layer=norm_layer,
                    sr_ratio=sr_ratios[1],
                )
                for i in range(depths[1])
            ]
        )
        self.norm2 = norm_layer(embed_dims[1])

        cur += depths[1]
        self.block3 = nn.ModuleList(
            [
                Block(
                    dim=embed_dims[2],
                    num_heads=num_heads[2],
                    mlp_ratio=mlp_ratios[2],
                    drop_path=dpr[cur + i],
                    norm_layer=norm_layer,
                    sr_ratio=sr_ratios[2],
                )
                for i in range(depths[2])
            ]
        )
        self.norm3 = norm_layer(embed_dims[2])

        cur += depths[2]
        self.block4 = nn.ModuleList(
            [
                Block(
                    dim=embed_dims[3],
                    num_heads=num_heads[3],
                    mlp_ratio=mlp_ratios[3],
                    drop_path=dpr[cur + i],
                    norm_layer=norm_layer,
                    sr_ratio=sr_ratios[3],
                )
                for i in range(depths[3])
            ]
        )
        self.norm4 = norm_layer(embed_dims[3])

        # classification head
        # self.head = nn.Linear(embed_dims[3], num_classes) if num_classes > 0 else nn.Identity()

    # def get_classifier(self):
    #     return self.head

    # def reset_classifier(self, num_classes):
    #     self.num_classes = num_classes
    #     self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward(self, x) -> List[torch.Tensor]:
        B = x.shape[0]
        outs: List[torch.Tensor] = []

        # stage 1
        x, x_proj_out = self.patch_embed1(x)
        _, _, H, W = x_proj_out.shape
        for i, blk in enumerate(self.block1):
            x = blk(x, H, W)
        x = self.norm1(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        # stage 2
        x, x_proj_out = self.patch_embed2(x)
        _, _, H, W = x_proj_out.shape
        for i, blk in enumerate(self.block2):
            x = blk(x, H, W)
        x = self.norm2(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        # stage 3
        x, x_proj_out = self.patch_embed3(x)
        _, _, H, W = x_proj_out.shape
        for i, blk in enumerate(self.block3):
            x = blk(x, H, W)
        x = self.norm3(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        # stage 4
        x, x_proj_out = self.patch_embed4(x)
        _, _, H, W = x_proj_out.shape
        for i, blk in enumerate(self.block4):
            x = blk(x, H, W)
        x = self.norm4(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        return outs


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H: int, W: int):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)

        return x


# class mit_b0(VisionTransformer):
#     def __init__(self, **kwargs):
#         super().__init__(
#             patch_size=4, embed_dims=[32, 64, 160, 256], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
#             norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[2, 2, 2, 2], sr_ratios=[8, 4, 2, 1],
#             drop_path_rate=0.1,embedding_dim = 256)


# class mit_b1(VisionTransformer):
#     def __init__(self, **kwargs):
#         super().__init__(
#             patch_size=4,
#             embed_dims=[64, 128, 320, 512],
#             num_heads=[1, 2, 5, 8],
#             mlp_ratios=[4, 4, 4, 4],
#             norm_layer=partial(nn.LayerNorm, eps=1e-6),
#             depths=[2, 2, 2, 2],
#             sr_ratios=[8, 4, 2, 1],
#             drop_path_rate=0.1,
#             embedding_dim=256,
#         )


# class mit_b2(VisionTransformer):
#     def __init__(self, **kwargs):
#         super().__init__(
#             patch_size=4,
#             embed_dims=[64, 128, 320, 512],
#             num_heads=[1, 2, 5, 8],
#             mlp_ratios=[4, 4, 4, 4],
#             norm_layer=partial(nn.LayerNorm, eps=1e-6),
#             depths=[3, 4, 6, 3],
#             sr_ratios=[8, 4, 2, 1],
#             drop_path_rate=0.1,
#         )


# class mit_b3(VisionTransformer):
#     def __init__(self, **kwargs):
#         super().__init__(
#             patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
#             norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 4, 18, 3], sr_ratios=[8, 4, 2, 1],
#             drop_path_rate=0.1)


class mit_b4(VisionTransformer):
    def __init__(self, **kwargs):
        super().__init__(
            patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
            norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 8, 27, 3], sr_ratios=[8, 4, 2, 1],
            drop_path_rate=0.1)


# class mit_b5(VisionTransformer):
#     def __init__(self, **kwargs):
#         super().__init__(
#             patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
#             norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 6, 40, 3], sr_ratios=[8, 4, 2, 1],
#             drop_path_rate=0.1)


class MLP(nn.Module):
    """
    Linear Embedding
    """

    def __init__(self, input_dim=2048, embed_dim=768):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x


class ConvModule(nn.Module):
    """A conv block that bundles conv/norm/activation layers."""

    def __init__(self,
        in_channels,
        out_channels,
        kernel_size,
        norm_cfg=None,
    ):
        super().__init__()

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=(1, 1),
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)

        self.activate = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activate(x)
        return x


class SegFormerHead(nn.Module):
    """
    SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers
    """

    def __init__(self, embedding_dim=768):
        super().__init__()
        # for b1 -- embedding_dim == 256
        # for b2 -- embedding_dim == 768

        self.feature_strides = [4, 8, 16, 32]
        self.in_channels = [64, 128, 320, 512]
        self.num_classes = 150  # for ADE20K dataset

        self.conv_seg = nn.Conv2d(128, self.num_classes, kernel_size=1)
        self.dropout = nn.Dropout2d(0.1)
        (
            c1_in_channels,
            c2_in_channels,
            c3_in_channels,
            c4_in_channels,
        ) = self.in_channels

        self.linear_c4 = MLP(input_dim=c4_in_channels, embed_dim=embedding_dim)
        self.linear_c3 = MLP(input_dim=c3_in_channels, embed_dim=embedding_dim)
        self.linear_c2 = MLP(input_dim=c2_in_channels, embed_dim=embedding_dim)
        self.linear_c1 = MLP(input_dim=c1_in_channels, embed_dim=embedding_dim)

        self.linear_fuse = ConvModule(
            in_channels=embedding_dim * 4,
            out_channels=embedding_dim,
            kernel_size=1,
            # norm_cfg=dict(type="SyncBN", requires_grad=True),
            norm_cfg=dict(type="BN", requires_grad=True),  # SyncBN Only for GPU, please use BN for CPU !!!
        )

        self.linear_pred = nn.Conv2d(embedding_dim, self.num_classes, kernel_size=1)

    def forward(self, inputs: List[torch.Tensor]):
        # len(inputs) --  4
        # inputs:  0  ---  [1, 64, 128, 128]
        # inputs:  1  ---  [1, 128, 64, 64]
        # inputs:  2  ---  [1, 320, 32, 32]
        # inputs:  3  ---  [1, 512, 16, 16]

        x = inputs  # len=4, 1/4,1/8,1/16,1/32
        c1, c2, c3, c4 = x
        # c1, c2, c3, c4:
        # [1, 64, 128, 128]
        # [1, 128, 64, 64]
        # [1, 320, 32, 32]
        # [1, 512, 16, 16]

        ############## MLP decoder on C1-C4 ###########
        n, _, h, w = c4.shape

        _c4 = self.linear_c4(c4).permute(0, 2, 1).reshape(n, -1, c4.shape[2], c4.shape[3])
        _c4 = F.interpolate(_c4, size=c1.size()[2:], mode="bilinear", align_corners=False)

        _c3 = self.linear_c3(c3).permute(0, 2, 1).reshape(n, -1, c3.shape[2], c3.shape[3])
        _c3 = F.interpolate(_c3, size=c1.size()[2:], mode="bilinear", align_corners=False)

        _c2 = self.linear_c2(c2).permute(0, 2, 1).reshape(n, -1, c2.shape[2], c2.shape[3])
        _c2 = F.interpolate(_c2, size=c1.size()[2:], mode="bilinear", align_corners=False)

        _c1 = self.linear_c1(c1).permute(0, 2, 1).reshape(n, -1, c1.shape[2], c1.shape[3])

        _c = self.linear_fuse(torch.cat([_c4, _c3, _c2, _c1], dim=1))
        x = self.dropout(_c)

        x = self.linear_pred(x)

        return x


class SegmentModel(nn.Module):
    """Encoder Decoder segmentors."""

    def __init__(self):
        super().__init__()
        # self.MAX_H = 1024
        # self.MAX_W = 1024
        # self.MAX_TIMES = 4
        # # GPU half model -- 4G, 120ms

        self.backbone = mit_b4()
        self.decode_head = SegFormerHead(self.backbone.embedding_dim)
        self.num_classes = self.decode_head.num_classes

        self.load_weights()
        self.half().eval()


    def load_weights(self, model_path="models/image_segment.pth"):
        cdir = os.path.dirname(__file__)
        checkpoint = model_path if cdir == "" else cdir + "/" + model_path
        sd = torch.load(checkpoint)
        if 'state_dict' in sd.keys():
            sd = sd['state_dict']
        self.load_state_dict(sd)


    def forward(self, x):
        if x.is_cuda:
            x = x.half()

        B, C, H, W = x.shape
        # x.size() -- ([1, 3, 960, 1280])

        # normalize first
        x = normalize(x, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

        f = self.backbone(x)
        seg_logit = self.decode_head(f)
        # len(f), f[0].size(), f[1].size(), f[2].size(), f[3].size()
        # (4, ([1, 64, 240, 320]),
        #     ([1, 128, 120, 160]),
        #     ([1, 320, 60, 80]),
        #     ([1, 512, 30, 40]))
        # seg_logit.size() -- ([1, 150, 240, 320])

        # seg_logit = F.interpolate(seg_logit, size=x.size()[2:], mode="bilinear", align_corners=False)

        seg_logit = F.interpolate(seg_logit, size=(H,W), mode="bilinear", align_corners=False)
        seg_logit = F.softmax(seg_logit, dim=1)

        mask = seg_logit.argmax(dim=1).unsqueeze(0)
        # mask.dtype -- int64, size() -- [1, 1, 960, 1280]

        return mask.clamp(0, self.num_classes) # ADE20K class number is 150


class SegmentLabel(nn.Module):
    '''Segment Semantic Label Remapping'''
    def __init__(self, min_ratio=0.02):
        super().__init__()
        self.min_ratio = min_ratio

        mapping_name = "models/ade20k_semantic_rel.npy"
        cdir = os.path.dirname(__file__)
        mapping_name = mapping_name if cdir == "" else cdir + "/" + mapping_name
        label_mapping = torch.from_numpy(np.load(mapping_name)).to(torch.int64)
        self.register_buffer("label_mapping", label_mapping)

        # (Pdb) self.label_mapping -- (150, 150)
        # tensor([[ 32,  25,  97, ...,  86,  86,  80],
        #        [ 97,  86,  82, ...,  97,  97,  97],
        #        [ 86,  97, 136, ...,  21,  80,  43],
        #        ...,
        #        [111, 118,  62, ..., 149,  54,  54],
        #        [118, 135, 118, ...,  54, 111, 118],
        #        [  0,   1,   2, ..., 147, 148, 149]])
        # torch.int64

    def cross_remapping(self, segment, guide_segment):
        # Example: style segment guide content segment
        unique_content_labels = torch.unique(segment)
        unique_guide_labels = torch.unique(guide_segment)

        new_segment = segment.clone()
        new_unique_content_labels = unique_content_labels.clone()

        for hole in new_unique_content_labels:
            new_hole = self.find_closest_label(hole, unique_guide_labels)
            new_unique_content_labels[unique_content_labels == hole] = new_hole

        for i, label in enumerate(new_unique_content_labels):
            new_segment[segment == label] = new_unique_content_labels[i]

        return new_segment


    def find_closest_label(self, label, guide_labels):
        # pdb.set_trace()
        candidate_sets = self.label_mapping[:, int(label)]
        for label in candidate_sets:
            if label in guide_labels: # OK we find 
                return label
        return label # Sorry we dont find, use original


    def forward(self, segment):
        # self_remapping -- Replace small hole with closest big one
        B, C, H, W = segment.size()
        min_pixels = max(int(H * W * self.min_ratio), 10)

        new_segment = segment.clone()
        unique_labels, unique_counts = torch.unique(segment, return_counts=True)
        small_holes = unique_labels[unique_counts < min_pixels]
        guide_labels = unique_labels[unique_counts >= min_pixels]

        new_unique_labels = unique_labels.clone()
        for hole in small_holes:
            new_hole = self.find_closest_label(hole, guide_labels)
            new_unique_labels[unique_labels == hole] = new_hole

        for i, label in enumerate(unique_labels):
            new_segment[segment == label] = new_unique_labels[i]

        return new_segment
