import torch
import torch.nn as nn
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

        B, C, H, W = content_feat.size()
        content_feat = content_feat.reshape(B, C, -1) # size() -- [1, 32, 811200]
        style_feat = style_feat.reshape(B, C, -1) # size() -- [1, 32, 921600]

        content_seg = content_seg.unsqueeze(0).unsqueeze(0)
        style_seg = style_seg.unsqueeze(0).unsqueeze(0)
        label_set, guide_labels = self.compute_label_info(content_seg, style_seg)

        for i in range(B):
            single_content_feat = content_feat[i]     # [C, H*W]
            single_style_feat = style_feat[i]   # [C, sH*sW]
            target_feature = single_content_feat.clone()   # [C, H*W]

            for label in label_set:
                if not guide_labels[int(label)]:
                    continue

                content_index = self.get_label_index(content_seg, label)
                style_index = self.get_label_index(style_seg, label)
                if content_index.size(0) < 1 or style_index.size(0) < 1:
                    continue

                masked_content_feat = torch.index_select(single_content_feat, 1, content_index)
                masked_style_feat = torch.index_select(single_style_feat, 1, style_index)
                whiten_feat = self.content_feat_whitening(masked_content_feat)
                color_feature = self.content_feat_coloring(whiten_feat, masked_style_feat)

                new_target_feature = target_feature.t() # size() -- [811200, 32]
                new_target_feature.index_copy_(0, content_index, color_feature.t())

                target_feature = new_target_feature.t()

            content_feat[i] = target_feature

        return content_feat.reshape(B, C, H, W)

    def cholesky_dec(self, conv, invert: bool=False):
        # conv.size() -- [32, 32]
        # torch.linalg.cholesky(A, *, upper=False, out=None)
        # try:
        #     L = torch.linalg.cholesky(conv)
        # except RuntimeError:
        #     # print("Warning: Cholesky Decomposition fails")
        #     iden = torch.eye(conv.shape[-1]).to(conv.device)
        #     eps = self.eps
        #     while True:
        #         try:
        #             conv = conv + iden * eps
        #             L = torch.linalg.cholesky(conv)
        #             break
        #         except RuntimeError:
        #             eps = eps+self.eps
        L = torch.linalg.cholesky(conv) # !!! onnx not support !!!

        # (L * L.t() - conv).abs().max() -- tensor(0.002681, device='cuda:0')
        if invert:
            L = torch.inverse(L) # onnx not support

        return L

    def content_feat_whitening(self, x):
        # x -- content features
        # x.size() -- [32, 167352]

        mean = torch.mean(x, -1).unsqueeze(-1).expand_as(x)
        x = x - mean
        # x.size() -- [32, 167352]

        conv = (x @ x.t()).div(x.shape[1] - 1)
        inv_L = self.cholesky_dec(conv, invert=True)
        # inv_L.size() -- [32, 32]
        whiten_x = inv_L @ x

        # U, S, V = torch.svd(conv)
        # S = (S + self.eps).sqrt()
        # # S = 1.0/S
        # ZCA = U * torch.diag(S) * U.t()
        # whiten_y = ZCA @ x
        # print("Max ABS: ", (whiten_x - whiten_y).abs().max())

        return whiten_x #size() -- [32, 167352]


    def content_feat_coloring(self, content_whiten_feat, style_feat):
        xs_mean = torch.mean(style_feat, -1)
        style_feat = style_feat - xs_mean.unsqueeze(-1).expand_as(style_feat)

        conv = (style_feat @ style_feat.t()).div(style_feat.shape[-1] - 1)
        # conv.size() -- [32, 32]
        Ls = self.cholesky_dec(conv, invert=False)

        coloring_cs = Ls @ content_whiten_feat # content_whiten_feat.size() -- [32, 167352]
        coloring_cs = coloring_cs + xs_mean.unsqueeze(-1).expand_as(coloring_cs)

        return coloring_cs # coloring_cs.size() -- [32, 167352]

    def compute_label_info(self, content_seg, style_seg):
        label_set = torch.unique(content_seg)
        # label_set -- tensor([ 2,  4,  9, 16, 21], device='cuda:0')

        max_label = torch.max(content_seg) + 1
        guide_labels = torch.zeros(max_label).to(torch.int64).to(label_set.device)

        for l in label_set:
            a = self.get_label_index(content_seg, l).size(0)
            b = self.get_label_index(style_seg, l).size(0)
            if a > 10 and b > 10 and a < (100 * b) and b < (100 * a):
                guide_labels[l] = 1

        # guide_labels -- 
        # Tensor([0., 0., 1., 0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1.])
        return label_set, guide_labels

    def get_label_index(self, segment, label):
        mask = torch.where(segment.flatten() == label)
        return mask[0]
