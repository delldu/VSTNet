import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from PIL import Image
import pdb

class cWCT(nn.Module):
    '''
    Cholesky decomposition based WCT    
    '''
    def __init__(self, eps=2e-5, use_double=False):
        super().__init__()
        self.eps = eps
        self.use_double = use_double

    def transfer(self, content_feat, style_feat, cmask=None, smask=None):
        if cmask is None or smask is None:
            return self._transfer(content_feat, style_feat)
        else:
            return self._transfer_seg(content_feat, style_feat, cmask, smask)

    def _transfer(self, content_feat, style_feat):
        """
        :param content_feat: [B, N, cH, cW]
        :param style_feat: [B, N, sH, sW]
        :return color_feat: [B, N, cH, cW]
        """
        B, N, cH, cW = content_feat.shape
        content_feat = content_feat.reshape(B, N, -1)
        style_feat = style_feat.reshape(B, N, -1)

        in_dtype = content_feat.dtype
        if self.use_double:
            pdb.set_trace()
            content_feat = content_feat.double()
            style_feat = style_feat.double()

        # whitening and coloring transforms
        whiten_feat = self.whitening(content_feat)
        color_feat = self.coloring(whiten_feat, style_feat)

        if self.use_double:
            color_feat = color_feat.to(in_dtype)

        return color_feat.reshape(B, N, cH, cW)

    def _transfer_seg(self, content_feat, style_feat, cmask, smask):
        """
        :param content_feat: [B, N, cH, cW]
        :param style_feat: [B, N, sH, sW]
        :param cmask: numpy [B, _, _]
        :param smask: numpy [B, _, _]
        :return color_feat: [B, N, cH, cW]
        """
        # cmask.shape -- (1, 672, 1200)
        # smask.shape -- (1, 720, 1280)

        B, N, cH, cW = content_feat.shape # (1, 32, 672, 1200)
        _, _, sH, sW = style_feat.shape # (720, 1280)
        content_feat = content_feat.reshape(B, N, -1) # [1, 32, 806400]
        style_feat = style_feat.reshape(B, N, -1) # [1, 32, 921600]

        in_dtype = content_feat.dtype
        if self.use_double:
            content_feat = content_feat.double()
            style_feat = style_feat.double()

        for i in range(B):
            label_set, label_indicator = self.compute_label_info(cmask[i], smask[i])
            resized_content_seg = cmask[i] # self.resize(cmask[i], cH, cW) # (672, 1200)
            resized_style_seg = smask[i] # self.resize(smask[i], sH, sW) # (720, 1280)

            # resized_content_seg = F.interpolate(cmask[i], size=(cH, cW), mode="bilinear", align_corners=False)
            # resized_style_seg = F.interpolate(smask[i], size=(sH, sW), mode="bilinear", align_corners=False)

            single_content_feat = content_feat[i]     # [N, cH*cW]
            single_style_feat = style_feat[i]   # [N, sH*sW]
            target_feature = single_content_feat.clone()   # [N, cH*cW]

            for label in label_set:
                if not label_indicator[label]:
                    continue

                # resized_content_seg.shape -- (672, 1200)
                content_index = self.get_index(resized_content_seg, label).to(single_content_feat.device)
                style_index = self.get_index(resized_style_seg, label).to(single_style_feat.device)
                if content_index is None or style_index is None:
                    continue

                # single_content_feat.size() -- [32, 806400]
                # content_index.size() -- [167352]
                masked_content_feat = torch.index_select(single_content_feat, 1, content_index)
                masked_style_feat = torch.index_select(single_style_feat, 1, style_index)
                whiten_feat = self.whitening(masked_content_feat)
                _target_feature = self.coloring(whiten_feat, masked_style_feat)

                new_target_feature = torch.transpose(target_feature, 1, 0)
                new_target_feature.index_copy_(0, content_index, torch.transpose(_target_feature, 1, 0))
                target_feature = torch.transpose(new_target_feature, 1, 0)

            content_feat[i] = target_feature
        color_feat = content_feat

        if self.use_double:
            color_feat = color_feat.to(in_dtype)

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
        # array [content_seg] shape: (672, 1200), min: 2, max: 21, mean: 11.180663
        # array [style_seg] shape: (720, 1280), min: 2, max: 21, mean: 10.575559

        if content_seg.size is False or style_seg.size is False:
            return

        max_label = np.max(content_seg) + 1
        label_set = np.unique(content_seg) # array([ 2,  4,  9, 16, 21], dtype=uint8)
        label_indicator = np.zeros(max_label) # shape -- (22,)

        for l in label_set:
            is_valid = lambda a, b: a > 10 and b > 10 and a / b < 100 and b / a < 100
            o_cont_mask = np.where(content_seg.reshape(content_seg.shape[0] * content_seg.shape[1]) == l)
            # torch.where(torch.from_numpy(content_seg.reshape(content_seg.shape[0] * content_seg.shape[1])) == 2)[0]
            # ==> tensor([7, 8, 9, ..., 244634, 245831, 245832])

            o_styl_mask = np.where(style_seg.reshape(style_seg.shape[0] * style_seg.shape[1]) == l)
            label_indicator[l] = is_valid(o_cont_mask[0].size, o_styl_mask[0].size)

        # (Pdb) self.label_indicator
        # array([0., 0., 1., 0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1.])

        return label_set, label_indicator

    def resize(self, img, H, W):
        size = (W, H)

        if len(img.shape) == 2:
            return np.array(Image.fromarray(img).resize(size, Image.NEAREST))
        else:
            return np.array(Image.fromarray(img, mode='RGB').resize(size, Image.NEAREST))

    def get_index(self, feat, label):
        mask = np.where(feat.reshape(feat.shape[0] * feat.shape[1]) == label)
        if mask[0].size <= 0:
            return None

        return torch.LongTensor(mask[0])

    def interpolation(self, content_feat, styl_feat_list, alpha_s_list, alpha_c=0.0):
        """
        :param content_feat: Tensor [B, N, cH, cW]
        :param styl_feat_list: List [Tensor [B, N, _, _], Tensor [B, N, _, _], ...]
        :param alpha_s_list: List [float, float, ...]
        :param alpha_c: float
        :return color_feat: Tensor [B, N, cH, cW]
        """
        assert len(styl_feat_list) == len(alpha_s_list)

        B, N, cH, cW = content_feat.shape
        content_feat = content_feat.reshape(B, N, -1)

        in_dtype = content_feat.dtype
        if self.use_double:
            content_feat = content_feat.double()

        c_mean = torch.mean(content_feat, -1)
        content_feat = content_feat - c_mean.unsqueeze(-1).expand_as(content_feat)

        cont_conv = (content_feat @ content_feat.transpose(-1, -2)).div(content_feat.shape[-1] - 1)  # interpolate Conv works well
        inv_Lc = self.cholesky_dec(cont_conv, invert=True)  # interpolate L seems to be slightly better

        whiten_c = inv_Lc @ content_feat

        # First interpolate between style_A, style_B, style_C, ...
        mix_Ls = torch.zeros_like(inv_Lc)   # [B, N, N]
        mix_s_mean = torch.zeros_like(c_mean)   # [B, N]
        for style_feat, alpha_s in zip(styl_feat_list, alpha_s_list):
            assert style_feat.shape[0] == B and style_feat.shape[1] == N
            style_feat = style_feat.reshape(B, N, -1)

            if self.use_double:
                style_feat = style_feat.double()

            s_mean = torch.mean(style_feat, -1)
            style_feat = style_feat - s_mean.unsqueeze(-1).expand_as(style_feat)

            styl_conv = (style_feat @ style_feat.transpose(-1, -2)).div(style_feat.shape[-1] - 1)  # interpolate Conv works well
            Ls = self.cholesky_dec(styl_conv, invert=False)  # interpolate L seems to be slightly better

            mix_Ls += Ls * alpha_s
            mix_s_mean += s_mean * alpha_s

        # Second interpolate between content and style_mix
        if alpha_c != 0.0:
            Lc = self.cholesky_dec(cont_conv, invert=False)
            mix_Ls = mix_Ls * (1-alpha_c) + Lc * alpha_c
            mix_s_mean = mix_s_mean * (1-alpha_c) + c_mean * alpha_c

        color_feat = mix_Ls @ whiten_c
        color_feat = color_feat + mix_s_mean.unsqueeze(-1).expand_as(color_feat)

        if self.use_double:
            color_feat = color_feat.to(in_dtype)

        return color_feat.reshape(B, N, cH, cW)


if __name__ == '__main__':
    # transfer
    c = torch.rand((2, 16, 512, 256))
    s = torch.rand((2, 16, 64, 128))

    cwct = cWCT(use_double=True)
    cs = cwct.transfer(c, s)
    print(cs.shape)


    # interpolation
    c = torch.rand((1, 16, 512, 256))
    s_list = [torch.rand((1, 16, 64, 128)) for _ in range(4)]
    alpha_s_list = [0.25 for _ in range(4)]     # interpolate between style_A, style_B, style_C, ...
    alpha_c = 0.5   # interpolate between content and style_mix if alpha_c!=0.0

    cwct = cWCT(use_double=True)
    cs = cwct.interpolation(c, s_list, alpha_s_list, alpha_c)
    print(cs.shape)
