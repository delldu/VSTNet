import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from PIL import Image
import pdb

class CWCT(nn.Module):
    '''
    Cholesky decomposition based WCT    
    '''
    def __init__(self, eps=2e-5):
        super().__init__()
        self.eps = eps

    def forward(self, content_feat, style_feat, content_seg, style_seg):
        # content_feat.size() -- [1, 32, 676, 1200]
        # style_feat.size() -- [1, 32, 720, 1280]
        # content_seg.size() -- [1, 1, 676, 1200]
        # style_seg.size() -- [1, 1, 720, 1280]

        B, N, cH, cW = content_feat.shape
        _, _, sH, sW = style_feat.shape
        content_feat = content_feat.reshape(B, N, -1) # size() -- [1, 32, 811200==676*1200]
        style_feat = style_feat.reshape(B, N, -1) # size() -- [1, 32, 921600==720*1280]
        content_seg = content_seg.unsqueeze(0)
        style_seg = style_seg.unsqueeze(0)

        for i in range(B):
            label_set, label_indicator = self.compute_label_info(content_seg[i], style_seg[i])

            resized_content_seg = content_seg[i]
            resized_style_seg = style_seg[i]

            single_content_feat = content_feat[i]     # [N, cH*cW]
            single_style_feat = style_feat[i]   # [N, sH*sW]
            target_feature = single_content_feat.clone()   # [N, cH*cW]

            for label in label_set:
                if not label_indicator[label]:
                    continue

                content_index = self.get_index(resized_content_seg, label)
                style_index = self.get_index(resized_style_seg, label)
                if content_index is None or style_index is None:
                    continue

                masked_content_feat = torch.index_select(single_content_feat, 1, content_index)
                masked_style_feat = torch.index_select(single_style_feat, 1, style_index)
                whiten_feat = self.whitening(masked_content_feat)
                _target_feature = self.coloring(whiten_feat, masked_style_feat)

                new_target_feature = torch.transpose(target_feature, 1, 0)
                new_target_feature.index_copy_(0, content_index, torch.transpose(_target_feature, 1, 0))
                target_feature = torch.transpose(new_target_feature, 1, 0)

            content_feat[i] = target_feature
        color_feat = content_feat

        return color_feat.reshape(B, N, cH, cW)

    def cholesky_dec(self, conv, invert=False):
        # conv.size() -- [32, 32]
        # torch.linalg.cholesky(A, *, upper=False, out=None)
        cholesky = torch.linalg.cholesky if torch.__version__ >= '1.8.0' else torch.cholesky
        try:
            L = cholesky(conv)
        except RuntimeError:
            pdb.set_trace()
            # print("Warning: Cholesky Decomposition fails")
            iden = torch.eye(conv.shape[-1]).to(conv.device)
            eps = self.eps
            while True:
                try:
                    conv = conv + iden * eps
                    L = cholesky(conv)
                    break
                except RuntimeError:
                    eps = eps+self.eps
        # (L * L.t() - conv).abs().max() -- tensor(0.002681, device='cuda:0')
        if invert:
            L = torch.inverse(L)
        return L.to(conv.dtype)

    def whitening(self, x):
        # x -- content features
        # x.size() -- [32, 167352]

        mean = torch.mean(x, -1)
        mean = mean.unsqueeze(-1).expand_as(x)
        # pdb.set_trace()

        x = x - mean
        # x.size() -- [32, 167352]
        conv = (x @ x.transpose(1, 0)).div(x.shape[1] - 1)
        inv_L = self.cholesky_dec(conv, invert=True)
        # inv_L.size() -- [32, 32]
        whiten_x = inv_L @ x

        return whiten_x #size() -- [32, 167352]


    def coloring(self, content_whiten_feat, style_feat):
        xs_mean = torch.mean(style_feat, -1)
        style_feat = style_feat - xs_mean.unsqueeze(-1).expand_as(style_feat)
        # pdb.set_trace()

        conv = (style_feat @ style_feat.transpose(-1, -2)).div(style_feat.shape[-1] - 1)
        # conv.size() -- [32, 32]
        Ls = self.cholesky_dec(conv, invert=False)

        coloring_cs = Ls @ content_whiten_feat # content_whiten_feat.size() -- [32, 167352]
        coloring_cs = coloring_cs + xs_mean.unsqueeze(-1).expand_as(coloring_cs)

        return coloring_cs # coloring_cs.size() -- [32, 167352]

    def compute_label_info(self, content_seg, style_seg):
        label_set = torch.unique(content_seg)
        # label_set -- tensor([ 2,  4,  9, 16, 21], device='cuda:0')

        max_label = torch.max(content_seg) + 1
        label_indicator = torch.zeros(max_label)
        # label_indicator -- tensor([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])

        for l in label_set:
            is_valid = lambda a, b: a > 10 and b > 10 and a / b < 100 and b / a < 100
            # o_cont_mask = torch.where(content_seg.reshape(content_seg.shape[0] * content_seg.shape[1]) == l)
            o_cont_mask = torch.where(content_seg.flatten() == l)


            # torch.where(torch.from_numpy(content_seg.reshape(content_seg.shape[0] * content_seg.shape[1])) == 2)[0]
            # ==> tensor([7, 8, 9, ..., 244634, 245831, 245832])
            # o_styl_mask = torch.where(style_seg.reshape(style_seg.shape[0] * style_seg.shape[1]) == l)
            o_styl_mask = torch.where(style_seg.flatten() == l)

            label_indicator[l] = is_valid(o_cont_mask[0].size(0), o_styl_mask[0].size(0))

        # (Pdb) self.label_indicator
        # array([0., 0., 1., 0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1.])

        return label_set, label_indicator

    def get_index(self, segment, label):
        mask = torch.where(segment.flatten() == label)
        return mask[0]
