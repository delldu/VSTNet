import torch
import torch.nn as nn
import todos
import pdb

class CWCT(nn.Module):
    '''
    Cholesky decomposition based WCT    
    '''
    def __init__(self, eps=2e-5):
        super().__init__()
        self.eps = eps

    def forward(self, c_feat, s_feat, c_mask, s_mask):
        # c_feat.size() -- [1, 32, 676, 1200]
        # s_feat.size() -- [1, 32, 720, 1280]
        # c_mask.size() -- [1, 1, 676, 1200]
        # s_mask.size() -- [1, 1, 720, 1280]

        B, C, H, W = c_feat.size()
        c_feat = c_feat.reshape(B, C, -1) # size() -- [1, 32, 811200]
        s_feat = s_feat.reshape(B, C, -1) # size() -- [1, 32, 921600]

        c_mask = c_mask.unsqueeze(0).unsqueeze(0)
        s_mask = s_mask.unsqueeze(0).unsqueeze(0)
        label_set, guide_labels = self.compute_label_info(c_mask, s_mask)
        # label_set -- tensor([ 2,  4,  9, 16, 21]
        # guide_labels -- [0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1] -- size() -- 22

        for i in range(B):
            single_content_feat = c_feat[i]     # [C, H*W]
            single_style_feat = s_feat[i]   # [C, sH*sW]
            target_feature = single_content_feat.clone()   # [C, H*W]

            for label in label_set:
                if not guide_labels[int(label)]:
                    continue

                content_index = self.get_label_index(c_mask, label)
                style_index = self.get_label_index(s_mask, label)

                # if content_index.size(0) < 1 or style_index.size(0) < 1:
                #     print(f"missing {label} .....................")
                #     continue

                print(f"label = {label} -----------------------------------------------")
                todos.debug.output_var("c_feat", single_content_feat)
                todos.debug.output_var("s_feat", single_style_feat)

                # single_content_feat.size() -- [32, 589824]
                # content_index.size() -- [122365]
                selected_c_feat = torch.index_select(single_content_feat, 1, content_index)
                # selected_c_feat.size() -- [32, 122365]
                selected_s_feat = torch.index_select(single_style_feat, 1, style_index)
                todos.debug.output_var("selected_c_feat", selected_c_feat)
                todos.debug.output_var("selected_s_feat", selected_s_feat)

                whiten_feat = self.content_feat_whitening(selected_c_feat)
                color_feature = self.content_feat_coloring(whiten_feat, selected_s_feat)
                todos.debug.output_var("whiten_feat", whiten_feat)
                todos.debug.output_var("color_feature", color_feature)

                print(f"-----------------------------------------------")

                # color_feature.size() -- [32, 122365]

                new_target_feature = target_feature.t() # size() -- [811200, 32]
                new_target_feature.index_copy_(0, content_index, color_feature.t())

                target_feature = new_target_feature.t()

            c_feat[i] = target_feature

        return c_feat.reshape(B, C, H, W)

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
        inv_L = self.cholesky_dec(conv, invert=True) # size() -- [32, 32]
        # inv_L.size() -- [32, 32]
        whiten_x = inv_L @ x

        todos.debug.output_var("conv", conv)
        todos.debug.output_var("inv_L", inv_L)
        todos.debug.output_var("whiten_x", whiten_x)


        # U, S, V = torch.svd(conv)
        # S = (S + self.eps).sqrt()
        # # S = 1.0/S
        # ZCA = U * torch.diag(S) * U.t() # size() -- [32, 32]
        # whiten_y = ZCA @ x
        # print("Max ABS: ", (whiten_x - whiten_y).abs().max()) # -- 20.505251

        return whiten_x #size() -- [32, 167352]


    def content_feat_coloring(self, c_whiten_feat, s_feat):
        s_mean = torch.mean(s_feat, -1)
        s_feat = s_feat - s_mean.unsqueeze(-1).expand_as(s_feat)

        conv = (s_feat @ s_feat.t()).div(s_feat.shape[-1] - 1)
        # conv.size() -- [32, 32]
        Ls = self.cholesky_dec(conv, invert=False)

        c_color_feat = Ls @ c_whiten_feat # c_whiten_feat.size() -- [32, 167352]
        c_color_feat = c_color_feat + s_mean.unsqueeze(-1).expand_as(c_color_feat)

        todos.debug.output_var("conv", conv)
        todos.debug.output_var("Ls", Ls)
        todos.debug.output_var("c_color_feat", c_color_feat)

        return c_color_feat # c_color_feat.size() -- [32, 167352]

    def compute_label_info(self, c_mask, s_mask):
        label_set = torch.unique(c_mask)
        # label_set -- tensor([ 2,  4,  9, 16, 21], device='cuda:0')

        max_label = torch.max(c_mask) + 1
        guide_labels = torch.zeros(max_label).to(torch.int64).to(label_set.device)

        for l in label_set:
            a = self.get_label_index(c_mask, l).size(0)
            b = self.get_label_index(s_mask, l).size(0)
            if a > 10 and b > 10 and a < (10 * b) and b < (10 * a):
                guide_labels[l] = 1
            else:
                print(f"missing {l} .....................")

        # guide_labels -- 
        # Tensor([0., 0., 1., 0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1.])
        return label_set, guide_labels

    def get_label_index(self, segment, label):
        mask = torch.where(segment.flatten() == label)
        return mask[0] # size() -- [122365]
