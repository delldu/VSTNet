#ifndef __SEGFORMER__H__
#define __SEGFORMER__H__
#include "ggml_engine.h"
#include "ggml_nn.h"

#pragma GCC diagnostic ignored "-Wformat-truncation"

#include <vector>

/*
 ConvModule(
  (conv): Conv2d(3072, 768, kernel_size=(1, 1), stride=(1, 1), bias=False)
  (bn): BatchNorm2d(768, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (activate): ReLU()
) */
// -----------------------------------------------------------------------
struct ConvModule {
    // network hparams
    
    struct Conv2d conv;
    struct BatchNorm2d bn; // dell_add

    void create_weight_tensors(struct ggml_context* ctx) {
        conv.in_channels = 768 * 4;
        conv.out_channels = 768;
        conv.kernel_size = {1, 1};
        // conv.stride = { 1, 1 };
        // conv.padding = { 0, 0 };
        // // conv.dilation = { 1, 1 };
        // conv.is_depthwise = false;
        conv.has_bias = false;
        conv.create_weight_tensors(ctx);

        bn.num_features = 768;
        bn.create_weight_tensors(ctx);
    }

    void setup_weight_names(const char *prefix) {
        char s[GGML_MAX_NAME];

        snprintf(s, sizeof(s), "%s%s", prefix, "conv.");
        conv.setup_weight_names(s);

        snprintf(s, sizeof(s), "%s%s", prefix, "bn.");
        bn.setup_weight_names(s);
    }

    ggml_tensor_t* forward(struct ggml_context* ctx, ggml_tensor_t* x) {
        x = conv.forward(ctx, x);
        x = bn.forward(ctx, x);
        x = ggml_relu(ctx, x);

        return x;
    }
};

/*
 MLP(
  (proj): Linear(in_features=512, out_features=768, bias=True)
) */
// -----------------------------------------------------------------------------------
struct MLP {
    int input_dim = 512;
    
    // network params
    struct Linear proj;

    void create_weight_tensors(struct ggml_context* ctx) {
        proj.in_features = input_dim;
        proj.out_features = 768;
        proj.has_bias = true;
        proj.create_weight_tensors(ctx);
    }

    void setup_weight_names(const char *prefix) {
        char s[GGML_MAX_NAME];

        snprintf(s, sizeof(s), "%s%s", prefix, "proj.");
        proj.setup_weight_names(s);
    }

    ggml_tensor_t* forward(struct ggml_context* ctx, ggml_tensor_t* x) {
        // B, C, H, W = x.size()
        // x = x.flatten(2).transpose(1, 2)
        // x = self.proj(x)
        // return x.permute(0, 2, 1)
        int W = (int)x->ne[0];
        int H = (int)x->ne[1];
        int C = (int)x->ne[2];
        int B = (int)x->ne[3];

        x = ggml_reshape_3d(ctx, x, W*H, C, B);
        x = ggml_cont(ctx, ggml_permute(ctx, x, 1, 0, 2, 3)); // [WH, C, B] -> [C, WH, B]
        x = proj.forward(ctx, x);
        x = ggml_cont(ctx, ggml_permute(ctx, x, 1, 0, 2, 3)); // [C, WH, B] -> [WH, C, B]
        return x;
    }
};

// -------------------------------------------------------------------------
struct SegFormerHead {
    // network hparams
    int num_classes = 150;

    // network params
    struct Conv2d conv_seg;
    struct MLP linear_c1;
    struct MLP linear_c2;
    struct MLP linear_c3;
    struct MLP linear_c4;
    struct ConvModule linear_fuse;
    struct Conv2d linear_pred;

    void create_weight_tensors(struct ggml_context* ctx) {
        // self.conv_seg = nn.Conv2d(128, self.num_classes, kernel_size=1)
        conv_seg.in_channels = 128;
        conv_seg.out_channels = num_classes;
        conv_seg.kernel_size = {1, 1};
        conv_seg.stride = { 1, 1 };
        conv_seg.padding = { 0, 0 };
        // conv_seg.dilation = { 1, 1 };
        // conv_seg.is_depthwise = false;
        // conv_seg.has_bias = true;
        conv_seg.create_weight_tensors(ctx);

        // 64, 128, 320, 512
        linear_c1.input_dim = 64;
        linear_c1.create_weight_tensors(ctx);
        linear_c2.input_dim = 128;
        linear_c2.create_weight_tensors(ctx);
        linear_c3.input_dim = 320;
        linear_c3.create_weight_tensors(ctx);
        linear_c4.input_dim = 512;
        linear_c4.create_weight_tensors(ctx);

        // self.linear_fuse = ConvModule(in_channels=768 * 4, out_channels=768)
        linear_fuse.create_weight_tensors(ctx);

        // self.linear_pred = nn.Conv2d(768, self.num_classes, kernel_size=1)
        linear_pred.in_channels = 768;
        linear_pred.out_channels = num_classes;
        linear_pred.kernel_size = {1, 1};
        // linear_pred.stride = { 1, 1 };
        // linear_pred.padding = { 0, 0 };
        // linear_pred.dilation = { 1, 1 };
        // linear_pred.is_depthwise = false;
        // linear_pred.has_bias = true;
        linear_pred.create_weight_tensors(ctx);
    }

    void setup_weight_names(const char *prefix) {
        char s[GGML_MAX_NAME];

        snprintf(s, sizeof(s), "%s%s", prefix, "conv_seg.");
        conv_seg.setup_weight_names(s);

        snprintf(s, sizeof(s), "%s%s", prefix, "linear_c1.");
        linear_c1.setup_weight_names(s);
        snprintf(s, sizeof(s), "%s%s", prefix, "linear_c2.");
        linear_c2.setup_weight_names(s);
        snprintf(s, sizeof(s), "%s%s", prefix, "linear_c3.");
        linear_c3.setup_weight_names(s);
        snprintf(s, sizeof(s), "%s%s", prefix, "linear_c4.");
        linear_c4.setup_weight_names(s);
        snprintf(s, sizeof(s), "%s%s", prefix, "linear_fuse.");
        linear_fuse.setup_weight_names(s);

        snprintf(s, sizeof(s), "%s%s", prefix, "linear_pred.");
        linear_pred.setup_weight_names(s);
    }

    ggml_tensor_t* forward(struct ggml_context* ctx, std::vector<ggml_tensor_t *> xlist) {
        // # inputs is tuple: len = 4
        // #     tensor [item] size: [1, 64, 240, 320], min: -4.25842, max: 4.218358, mean: 0.014021
        // #     tensor [item] size: [1, 128, 120, 160], min: -6.090078, max: 4.901278, mean: 0.02357
        // #     tensor [item] size: [1, 320, 60, 80], min: -5.592515, max: 4.761344, mean: -0.002071
        // #     tensor [item] size: [1, 512, 30, 40], min: -6.624208, max: 8.50036, mean: 0.025605
        ggml_tensor_t *x1 = xlist[0];
        ggml_tensor_t *x2 = xlist[1];
        ggml_tensor_t *x3 = xlist[2];
        ggml_tensor_t *x4 = xlist[3];

        int W1 = (int)x1->ne[0];
        int H1 = (int)x1->ne[1];
        // int C1 = (int)x1->ne[2];
        // int B1 = (int)x1->ne[3];

        int W4 = (int)x4->ne[0];
        int H4 = (int)x4->ne[1];
        // int C4 = (int)x4->ne[2];
        int B4 = (int)x4->ne[3];

        // _c4 = self.linear_c4(c4).reshape(B4, -1, H4, W4)
        // _c4 = F.interpolate(_c4, size=(H1, W1), mode="bilinear", align_corners=False)
        // x4
        {
            x4 = linear_c4.forward(ctx, x4);
            x4 = ggml_reshape_4d(ctx, x4, W4, H4, -1, B4);
            x4 = ggml_interpolate(ctx, x4, 0, W1); 
            x4 = ggml_interpolate(ctx, x4, 1, H1); 
        }

        // _c3 = self.linear_c3(c3).reshape(B4, -1, H3, W3)
        // _c3 = F.interpolate(_c3, size=(H1, W1), mode="bilinear", align_corners=False)
        // x3
        {
            int W3 = (int)x3->ne[0];
            int H3 = (int)x3->ne[1];
            x3 = linear_c3.forward(ctx, x3);
            x3 = ggml_reshape_4d(ctx, x3, W3, H3, -1, B4);
            x3 = ggml_interpolate(ctx, x3, 0, W1); 
            x3 = ggml_interpolate(ctx, x3, 1, H1); 
        }

        // _c2 = self.linear_c2(c2).reshape(B4, -1, H2, W2)
        // _c2 = F.interpolate(_c2, size=(H1, W1), mode="bilinear", align_corners=False)
        // x2
        {
            int W2 = (int)x2->ne[0];
            int H2 = (int)x2->ne[1];
            x2 = linear_c2.forward(ctx, x2);
            x2 = ggml_reshape_4d(ctx, x2, W2, H2, -1, B4);
            x2 = ggml_interpolate(ctx, x2, 0, W1); 
            x2 = ggml_interpolate(ctx, x2, 1, H1); 
        }

        // _c1 = self.linear_c1(c1).reshape(B4, -1, H1, W1)
        // x1
        {
            x1 = linear_c1.forward(ctx, x1);
            x1 = ggml_reshape_4d(ctx, x1, W1, H1, -1, B4);
        }

        ggml_tensor_t *x = ggml_concat(ctx, x4, x3, 2/*dim on channels*/);
        x = ggml_concat(ctx, x, x2, 2/*dim on channels*/);
        x = ggml_concat(ctx, x, x1, 2/*dim on channels*/);

        x = linear_fuse.forward(ctx, x);
        x = linear_pred.forward(ctx, x);        

        // tensor [x] size: [1, 150, 240, 256], min: -7.343958, max: -2.939925, mean: -4.71556

        return x;
    }
};

/*
 DWConv(
  (dwconv): Conv2d(2048, 2048, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=2048)
) */

// !!!-----------------------------------------------------------------------------------------
struct DWConv {
    int dim = 768;

    // network params
    struct Conv2d dwconv;

    void create_weight_tensors(struct ggml_context* ctx) {
        // self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)
        dwconv.in_channels = dim;
        dwconv.out_channels = dim;
        dwconv.kernel_size = {3, 3};
        dwconv.stride = { 1, 1 };
        dwconv.padding = { 1, 1 };
        // dwconv.dilation = { 1, 1 };
        dwconv.is_depthwise = true;
        dwconv.has_bias = true;
        dwconv.create_weight_tensors(ctx);
    }

    void setup_weight_names(const char *prefix) {
        char s[GGML_MAX_NAME];
        snprintf(s, sizeof(s), "%s%s", prefix, "dwconv.");
        dwconv.setup_weight_names(s);
    }

    ggml_tensor_t* forward(struct ggml_context* ctx, ggml_tensor_t* x, int H, int W) {
        // B, N, C = x.shape
        // x = x.transpose(1, 2).view(B, C, H, W)
        // x = self.dwconv(x)
        // x = x.flatten(2).transpose(1, 2)
        // return x

        int B = (int)x->ne[2];
        int HW = (int)x->ne[1];
        int C = (int)x->ne[0];
        x = ggml_cont(ctx, ggml_permute(ctx, x, 1, 0, 2, 3)); // [C, HW, B] -> [HW, C, B]
        x = ggml_reshape_4d(ctx, x, W, H, C, B);
        x = dwconv.forward(ctx, x);
        x = ggml_reshape_3d(ctx, x, W*H, C, B);
        x = ggml_cont(ctx, ggml_permute(ctx, x, 1, 0, 2, 3)); // [HW, C, B] -> [C, HW, B]

        // tensor [x] size: [1, 960, 2048], min: -2.741753, max: 1.693425, mean: -0.495102
        return x;
    }
};

/*
 BlockMlp(
  (fc1): Linear(in_features=512, out_features=2048, bias=True)
  (dwconv): DWConv(
    (dwconv): Conv2d(2048, 2048, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=2048)
  )
  (act): GELU(approximate='none')
  (fc2): Linear(in_features=2048, out_features=512, bias=True)
) */

// --------------------------------------------------------------------------------
struct BlockMlp {
    int in_features;
    int hidden_features;

    // network params
    struct Linear fc1;
    struct DWConv dwconv;
    struct Linear fc2;

    void create_weight_tensors(struct ggml_context* ctx) {
        fc1.in_features = in_features;
        fc1.out_features = hidden_features;
        fc1.create_weight_tensors(ctx);

        dwconv.dim = hidden_features;
        dwconv.create_weight_tensors(ctx);

        fc2.in_features = hidden_features;
        fc2.out_features = in_features;
        fc2.create_weight_tensors(ctx);
    }

    void setup_weight_names(const char *prefix) {
        char s[GGML_MAX_NAME];

        snprintf(s, sizeof(s), "%s%s", prefix, "fc1.");
        fc1.setup_weight_names(s);

        snprintf(s, sizeof(s), "%s%s", prefix, "dwconv.");
        dwconv.setup_weight_names(s);

        snprintf(s, sizeof(s), "%s%s", prefix, "fc2.");
        fc2.setup_weight_names(s);
    }

    ggml_tensor_t* forward(struct ggml_context* ctx, ggml_tensor_t* x, int H, int W) {
        // x = self.fc1(x)
        // x = self.dwconv(x, H, W)
        // x = self.act(x)
        // x = self.fc2(x)
        // return x

        x = fc1.forward(ctx, x);
        x = dwconv.forward(ctx, x, H, W);
        x = ggml_gelu(ctx, x);
        x = fc2.forward(ctx, x);

        // tensor [x] size: [1, 960, 512], min: -21.298813, max: 18.067135, mean: -0.061462
    	return x;
    }
};

// -------------------------------------------------------------------------------
struct Attention {
    // network hparams
    int dim = 512;
    int num_heads = 8;
    int sr_ratio = 1;

    float scale = 0.125;

    // network params
    struct Linear q;
    struct Linear kv;
    struct Linear proj;

    struct Conv2d sr;
    struct LayerNorm norm;

    void create_weight_tensors(struct ggml_context* ctx) {
        q.in_features = dim;
        q.out_features = dim;
        q.has_bias = true;
        q.create_weight_tensors(ctx);

        kv.in_features = dim;
        kv.out_features = 2*dim;
        kv.has_bias = true;
        kv.create_weight_tensors(ctx);

        proj.in_features = dim;
        proj.out_features = dim;
        proj.has_bias = true;
        proj.create_weight_tensors(ctx);

        if (sr_ratio > 1) {
            sr.in_channels = dim;
            sr.out_channels = dim;
            sr.kernel_size = { sr_ratio, sr_ratio };
            sr.stride = { sr_ratio, sr_ratio };
            // sr.padding = { 0, 0 };
            // sr.dilation = { 1, 1 };
            // sr.is_depthwise = false;
            // sr.has_bias = true;
            sr.create_weight_tensors(ctx);

            norm.normalized_shape = dim;
            norm.create_weight_tensors(ctx);
        }
    }

    void setup_weight_names(const char *prefix) {
        char s[GGML_MAX_NAME];

        snprintf(s, sizeof(s), "%s%s", prefix, "q.");
        q.setup_weight_names(s);

        snprintf(s, sizeof(s), "%s%s", prefix, "kv.");
        kv.setup_weight_names(s);

        snprintf(s, sizeof(s), "%s%s", prefix, "proj.");
        proj.setup_weight_names(s);

        if (sr_ratio > 1) {
            snprintf(s, sizeof(s), "%s%s", prefix, "sr.");
            sr.setup_weight_names(s);

            snprintf(s, sizeof(s), "%s%s", prefix, "norm.");
            norm.setup_weight_names(s);
        }
    }

    ggml_tensor_t* forward(struct ggml_context* ctx, ggml_tensor_t* x, int H, int W) {
        // x -- [C, HW, B]
        int B = (int) x->ne[2];
        int HW = (int) x->ne[1];
        int C = (int) x->ne[0];
        GGML_ASSERT(HW == H*W);

        ggml_tensor_t *q_x = q.forward(ctx, x);
        q_x = ggml_reshape_4d(ctx, q_x, C/num_heads, num_heads, HW, B);
        q_x = ggml_cont(ctx, ggml_permute(ctx, q_x, 0, 2, 1, 3));
        // [C/num_heads, num_heads, HW, B] -> [C/num_heads, HW, num_heads, B]

        if (sr_ratio > 1) {
            x = ggml_cont(ctx, ggml_permute(ctx, x, 1, 0, 2, 3));
            x = ggml_reshape_4d(ctx, x, W, H, C, B);
            // [C, HW, B] -> [HW, C, B] -> [W, H, C, B]

            x = sr.forward(ctx, x);
            x = ggml_reshape_4d(ctx, x, -1, C, B, 1);
            x = ggml_cont(ctx, ggml_permute(ctx, x, 1, 0, 2, 3));
            // [W, H, C, B]->[HW, C, B] -> [C, HW, B]

            x = norm.forward(ctx, x);
        }
        ggml_tensor_t *kv_x = kv.forward(ctx, x);
        kv_x = ggml_reshape_4d(ctx, kv_x, C/num_heads, num_heads, -1, B);
        kv_x = ggml_cont(ctx, ggml_permute(ctx, kv_x, 0, 2, 1, 3));
        // [C/num_heads, num_heads, HW', B] -> [C/num_heads, HW', num_heads, B]

        int N2 = (int)kv_x->ne[1]; // dim 1
        ggml_tensor_t *k_x = ggml_nn_slice(ctx, kv_x, 1 /*dim*/, 0, N2, 2/*step*/);
        ggml_tensor_t *v_x = ggml_nn_slice(ctx, kv_x, 1 /*dim*/, 1, N2, 2/*step*/);

        ggml_tensor_t *attn = ggml_nn_mul_mat(ctx, q_x, ggml_transpose(ctx, k_x));
        // [HW', HW, num_heads, B]
        attn = ggml_scale(ctx, attn, scale);

        // attn = attn.softmax(dim=-1)
        attn = ggml_soft_max(ctx, attn);
        // ---------------------------------------------------------------------------------
        // torch: x = (attn @ v).transpose(1, 2).reshape(B, HW, C)
        x = ggml_nn_mul_mat(ctx, attn, v_x);
        x = ggml_cont(ctx, ggml_permute(ctx, x, 0, 2, 1, 3));
        x = ggml_reshape_3d(ctx, x, C, HW, B);
        // [C/num_heads, HW, num_heads, B] -> [C/num_heads, num_heads, HW, B]->[C, HW, B]

        x = proj.forward(ctx, x);
        GGML_ASSERT(C == (int)x->ne[0] && HW == (int)x->ne[1] && B == x->ne[2]);

        return x; // [C, HW, B]
    }
};

struct Block {
    int dim = 64;
    int num_heads = 1;
    int sr_ratio = 8;

    // network params
    struct LayerNorm norm1;
    struct Attention attn;
    struct LayerNorm norm2;
    struct BlockMlp mlp;

    void create_weight_tensors(struct ggml_context* ctx) {
        norm1.normalized_shape = dim;
        norm1.create_weight_tensors(ctx);

        attn.dim = dim;
        attn.num_heads = num_heads;
        attn.sr_ratio = sr_ratio;
        attn.create_weight_tensors(ctx);

        norm2.normalized_shape = dim;
        norm2.create_weight_tensors(ctx);

        mlp.in_features = dim;
        mlp.hidden_features = dim * 4;
        mlp.create_weight_tensors(ctx);
    }

    void setup_weight_names(const char *prefix) {
        char s[GGML_MAX_NAME];

        snprintf(s, sizeof(s), "%s%s", prefix, "norm1.");
        norm1.setup_weight_names(s);

        snprintf(s, sizeof(s), "%s%s", prefix, "attn.");
        attn.setup_weight_names(s);

        snprintf(s, sizeof(s), "%s%s", prefix, "norm2.");
        norm2.setup_weight_names(s);

        snprintf(s, sizeof(s), "%s%s", prefix, "mlp.");
        mlp.setup_weight_names(s);
    }

    ggml_tensor_t* forward(struct ggml_context* ctx, ggml_tensor_t* x, int H, int W) {
        // x = x + self.attn(self.norm1(x), H, W) 
        // x = x + self.mlp(self.norm2(x), H, W)
        // return x

        // x -- [C, HW, B]
        int B = (int) x->ne[2];
        int HW = (int) x->ne[1];
        int C = (int) x->ne[0];

        // x --- [C, HW, B]
        ggml_tensor_t *n;
        n = norm1.forward(ctx, x);
        n = attn.forward(ctx, n, H, W);
        x = ggml_add(ctx, x, n);
        // -----------------------------
        n = norm2.forward(ctx, x);
        n = mlp.forward(ctx, n, H, W);
        x = ggml_add(ctx, x, n);

        // x --- [C, HW, B]
        GGML_ASSERT(C == (int)x->ne[0] && HW == (int)x->ne[1] && B == x->ne[2]);

        return x;
    }
};

/*
 OverlapPatchEmbed(
  (proj): Conv2d(320, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
  (norm): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
) */

// ------------------------------------------------------------------------
struct OverlapPatchEmbed {
    // network hparams
    int in_chans = 3;
    int embed_dim = 768;
    int stride = 4;
    int patch_size = 7;

    struct Conv2d proj;
    struct LayerNorm norm;

    void create_weight_tensors(struct ggml_context* ctx) {
        proj.in_channels = in_chans;
        proj.out_channels = embed_dim;
        proj.kernel_size = { patch_size, patch_size };
        proj.stride = { stride, stride };
        proj.padding = { patch_size/2, patch_size/2 };
        // proj.dilation = { 1, 1 };
        // proj.is_depthwise = false;
        // proj.has_bias = true;
        proj.create_weight_tensors(ctx);

        norm.normalized_shape = embed_dim;
        norm.create_weight_tensors(ctx);
    }

    void setup_weight_names(const char *prefix) {
        char s[GGML_MAX_NAME];

        snprintf(s, sizeof(s), "%s%s", prefix, "proj.");
        proj.setup_weight_names(s);

        snprintf(s, sizeof(s), "%s%s", prefix, "norm.");
        norm.setup_weight_names(s);
    }

    ggml_tensor_t* forward(struct ggml_context* ctx, ggml_tensor_t* x, int *H, int *W) {
        // x = self.proj(x)
        // proj_out = x
        // x = x.flatten(2).transpose(1, 2) # (B, C, HW) -> (B, HW, C)
        // x = self.norm(x)
        // return (x, proj_out)
        x = proj.forward(ctx, x);

        *W = (int)x->ne[0];
        *H = (int)x->ne[1];
        int C = (int)x->ne[2];
        int B = (int)x->ne[3];
        x = ggml_cont(ctx, ggml_reshape_3d(ctx, x, (*W)*(*H), C, B));
        x = ggml_cont(ctx, ggml_permute(ctx, x, 1, 0, 2, 3)); // (HW, C, B) -> (C, HW, B)
        x = norm.forward(ctx, x); // [1, 3840, 320]

        // tensor [x] size: [1, 61440, 64], min: -0.880841, max: 0.975106, mean: -0.016517
        return x;
    }
};


struct VisionTransformer {
    // network hparams
    int embed_dims[4] = {64, 128, 320, 512};
    int num_heads[4] = {1, 2, 5, 8};
    int depths[4] = {3, 8, 27, 3};
    int sr_ratios[4] = {8, 4, 2, 1};

    // network params
    struct OverlapPatchEmbed patch_embed1;
    struct OverlapPatchEmbed patch_embed2;
    struct OverlapPatchEmbed patch_embed3;
    struct OverlapPatchEmbed patch_embed4;

    // depth [3, 8, 27, 3]
    struct Block block1[3];
    struct LayerNorm norm1;

    struct Block block2[8];
    struct LayerNorm norm2;

    struct Block block3[27];
    struct LayerNorm norm3;

    struct Block block4[3];
    struct LayerNorm norm4;

    void create_weight_tensors(struct ggml_context* ctx) {
        // embed_dims=[64, 128, 320, 512]
        // self.patch_embed1 = OverlapPatchEmbed(patch_size=7, stride=4, in_chans=3, embed_dim=embed_dims[0])
        // self.patch_embed2 = OverlapPatchEmbed(patch_size=3, stride=2, in_chans=embed_dims[0], embed_dim=embed_dims[1])
        // self.patch_embed3 = OverlapPatchEmbed(patch_size=3, stride=2, in_chans=embed_dims[1], embed_dim=embed_dims[2])
        // self.patch_embed4 = OverlapPatchEmbed(patch_size=3, stride=2, in_chans=embed_dims[2], embed_dim=embed_dims[3])

        patch_embed1.in_chans = 3;
        patch_embed1.embed_dim = 64;
        patch_embed1.stride = 4;
        patch_embed1.patch_size = 7;
        patch_embed1.create_weight_tensors(ctx);

        patch_embed2.in_chans = 64;
        patch_embed2.embed_dim = 128;
        patch_embed2.stride = 2;
        patch_embed2.patch_size = 3;
        patch_embed2.create_weight_tensors(ctx);

        patch_embed3.in_chans = 128;
        patch_embed3.embed_dim = 320;
        patch_embed3.stride = 2;
        patch_embed3.patch_size = 3;
        patch_embed3.create_weight_tensors(ctx);

        patch_embed4.in_chans = 320;
        patch_embed4.embed_dim = 512;
        patch_embed4.stride = 2;
        patch_embed4.patch_size = 3;
        patch_embed4.create_weight_tensors(ctx);

        for (int i = 0; i < 3; i++) {
            block1[i].dim = embed_dims[0];
            block1[i].num_heads = num_heads[0];
            block1[i].sr_ratio = sr_ratios[0];

            block1[i].create_weight_tensors(ctx);
        }
        // block1.create_weight_tensors(ctx);
        norm1.normalized_shape = embed_dims[0];
        norm1.eps = 1e-6;
        norm1.create_weight_tensors(ctx);

        for (int i = 0; i < 8; i++) {
            block2[i].dim = embed_dims[1];
            block2[i].num_heads = num_heads[1];
            block2[i].sr_ratio = sr_ratios[1];

            block2[i].create_weight_tensors(ctx);
        }
        // block2.create_weight_tensors(ctx);
        norm2.normalized_shape = embed_dims[1];
        norm2.eps = 1e-6;
        norm2.create_weight_tensors(ctx);

        for (int i = 0; i < 27; i++) {
            block3[i].dim = embed_dims[2];
            block3[i].num_heads = num_heads[2];
            block3[i].sr_ratio = sr_ratios[2];

            block3[i].create_weight_tensors(ctx);
        }
        // block3.create_weight_tensors(ctx);
        norm3.normalized_shape = embed_dims[2];
        norm3.eps = 1e-6;
        norm3.create_weight_tensors(ctx);

        for (int i = 0; i < 3; i++) {
            block4[i].dim = embed_dims[3];
            block4[i].num_heads = num_heads[3];
            block4[i].sr_ratio = sr_ratios[3];

            block4[i].create_weight_tensors(ctx);
        }
        // block4.create_weight_tensors(ctx);
        norm4.normalized_shape = embed_dims[3];
        norm4.eps = 1e-6;
        norm4.create_weight_tensors(ctx);
    }

    void setup_weight_names(const char *prefix) {
        char s[GGML_MAX_NAME];
        snprintf(s, sizeof(s), "%s%s", prefix, "patch_embed1.");
        patch_embed1.setup_weight_names(s);
        snprintf(s, sizeof(s), "%s%s", prefix, "patch_embed2.");
        patch_embed2.setup_weight_names(s);
        snprintf(s, sizeof(s), "%s%s", prefix, "patch_embed3.");
        patch_embed3.setup_weight_names(s);
        snprintf(s, sizeof(s), "%s%s", prefix, "patch_embed4.");
        patch_embed4.setup_weight_names(s);

        for (int i = 0; i < 3; i++) {
            // snprintf(s, sizeof(s), "%s%s", prefix, "block1.0.");
            snprintf(s, sizeof(s), "%sblock1.%d.", prefix, i);
            block1[i].setup_weight_names(s);
        }
        snprintf(s, sizeof(s), "%s%s", prefix, "norm1.");
        norm1.setup_weight_names(s);

        for (int i = 0; i < 8; i++) {
            snprintf(s, sizeof(s), "%sblock2.%d.", prefix, i);
            block2[i].setup_weight_names(s);
        }
        snprintf(s, sizeof(s), "%s%s", prefix, "norm2.");
        norm2.setup_weight_names(s);

        for (int i = 0; i < 27; i++) {
            snprintf(s, sizeof(s), "%sblock3.%d.", prefix, i);
            block3[i].setup_weight_names(s);
        }
        snprintf(s, sizeof(s), "%s%s", prefix, "norm3.");
        norm3.setup_weight_names(s);

        for (int i = 0; i < 3; i++) {
            snprintf(s, sizeof(s), "%sblock4.%d.", prefix, i);
            block4[i].setup_weight_names(s);
        }
        snprintf(s, sizeof(s), "%s%s", prefix, "norm4.");
        norm4.setup_weight_names(s);
    }

    std::vector<ggml_tensor_t *> forward(struct ggml_context* ctx, ggml_tensor_t* x) {
        std::vector<ggml_tensor_t *> xlist;
        // B = x.shape[0]

        // # stage 1
        int B = (int)x->ne[3];
        int C = (int)x->ne[2];
        int H = (int)x->ne[1];
        int W = (int)x->ne[0];

        // x, x_proj_out = self.patch_embed1(x)
        // _, _, H, W = x_proj_out.shape
        // for i, blk in enumerate(self.block1):
        //     x = blk(x, H, W)
        // x = self.norm1(x)
        // x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        // x1 = x
        x = patch_embed1.forward(ctx, x, &H, &W);
        // tensor [x] size: [1, 61440, 64], min: -2.743275, max: 2.450053, mean: -0.018798
        for (int i = 0; i < 3; i++) {
            x = block1[i].forward(ctx, x, H, W);
        }
        // tensor [x] size: [1, 61440, 64], min: -5.960706, max: 3.477771, mean: -0.034079

        x = norm1.forward(ctx, x);
        x = ggml_reshape_4d(ctx, x, -1/*C*/, W, H, B);
        x = ggml_cont(ctx, ggml_permute(ctx, x, 2, 0, 1, 3)); // [C, W, H, B] -> [W, H, C, B]
        xlist.push_back(x);

        // # stage 2
        // x, x_proj_out = self.patch_embed2(x)
        // _, _, H, W = x_proj_out.shape
        // for i, blk in enumerate(self.block2):
        //     x = blk(x, H, W)
        // x = self.norm2(x)
        // x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        // x2 = x
        x = patch_embed2.forward(ctx, x, &H, &W);
        for (int i = 0; i < 8; i++) {
            x = block2[i].forward(ctx, x, H, W);
        }
        x = norm2.forward(ctx, x);
        x = ggml_reshape_4d(ctx, x, -1, W, H, B);
        x = ggml_cont(ctx, ggml_permute(ctx, x, 2, 0, 1, 3)); // [C, W, H, B] -> [W, H, C, B]
        xlist.push_back(x);

        // # stage 3
        // x, x_proj_out = self.patch_embed3(x)
        // _, _, H, W = x_proj_out.shape
        // for i, blk in enumerate(self.block3):
        //     x = blk(x, H, W)
        // x = self.norm3(x)
        // x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        // x3 = x
        x = patch_embed3.forward(ctx, x, &H, &W);
        for (int i = 0; i < 27; i++) {
            x = block3[i].forward(ctx, x, H, W);
        }
        x = norm3.forward(ctx, x);
        x = ggml_reshape_4d(ctx, x, -1/*C*/, W, H, B);
        x = ggml_cont(ctx, ggml_permute(ctx, x, 2, 0, 1, 3)); // [C, W, H, B] -> [W, H, C, B]
        xlist.push_back(x);

        // # stage 4
        // x, x_proj_out = self.patch_embed4(x)
        // _, _, H, W = x_proj_out.shape
        // for i, blk in enumerate(self.block4):
        //     x = blk(x, H, W)
        // x = self.norm4(x)
        // x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        // x4 = x

        x = patch_embed4.forward(ctx, x, &H, &W);
        for (int i = 0; i < 3; i++) {
            x = block4[i].forward(ctx, x, H, W);
        }
        x = norm4.forward(ctx, x); // (C, HW, B)
        x = ggml_reshape_4d(ctx, x, -1/*C*/, W, H, B);
        x = ggml_cont(ctx, ggml_permute(ctx, x, 2, 0, 1, 3)); // [C, W, H, B] -> [W, H, C, B]
        xlist.push_back(x);

        // # tensor [x] size: [1, 3, 960, 1024], min: -2.117904, max: 2.64, mean: 0.034037
        // # tensor [x1] size: [1, 64, 240, 256], min: -4.195707, max: 4.108228, mean: 0.012766
        // # tensor [x2] size: [1, 128, 120, 128], min: -6.148735, max: 5.365499, mean: 0.023484
        // # tensor [x3] size: [1, 320, 60, 64], min: -5.416265, max: 4.744856, mean: -0.002299
        // # tensor [x4] size: [1, 512, 30, 32], min: -6.56222, max: 7.764719, mean: 0.025833
    	return xlist;
    }
};


struct SegmentModel : GGMLNetwork {
    // network hparams
    int MAX_H = 1024;
    int MAX_W = 2048;
    int MAX_TIMES = 4;
    int num_classes = 150;

    // network params
    struct Normalize normalize;
    struct VisionTransformer backbone;
    struct SegFormerHead decode_head;

    size_t get_graph_size()
    {
        return GGML_DEFAULT_GRAPH_SIZE * 4; // 2048 * 4
    }

    void create_weight_tensors(struct ggml_context* ctx) {
        backbone.create_weight_tensors(ctx);
        decode_head.create_weight_tensors(ctx);
    }

    void setup_weight_names(const char *prefix) {
        char s[GGML_MAX_NAME];
        snprintf(s, sizeof(s), "%s%s", prefix, "backbone.");
        backbone.setup_weight_names(s);
        snprintf(s, sizeof(s), "%s%s", prefix, "decode_head.");
        decode_head.setup_weight_names(s);
    }

    ggml_tensor_t* forward(ggml_context_t* ctx, int argc, ggml_tensor_t* argv[]) {
        GGML_UNUSED(argc);
        ggml_tensor_t* x = argv[0];

        int W = (int)x->ne[0];
        int H = (int)x->ne[1];
        // int C = (int)x->ne[2];
        // int B = (int)x->ne[3];
        int r_pad = (MAX_TIMES - (W % MAX_TIMES)) % MAX_TIMES;
        int b_pad = (MAX_TIMES - (H % MAX_TIMES)) % MAX_TIMES;
        x = ggml_replication_pad2d(ctx, x, 0, r_pad, 0, b_pad);

        x = normalize.forward(ctx, x);
        // tensor [x] size: [1, 3, 960, 1024], min: -2.117904, max: 2.64, mean: 0.034037

        std::vector<ggml_tensor_t *>xlist = backbone.forward(ctx, x);
        // f is tuple: len = 4
        //     tensor [item] size: [1, 64, 240, 256], min: -4.195707, max: 4.108228, mean: 0.012766
        //     tensor [item] size: [1, 128, 120, 128], min: -6.148735, max: 5.365499, mean: 0.023484
        //     tensor [item] size: [1, 320, 60, 64], min: -5.416265, max: 4.744856, mean: -0.002299
        //     tensor [item] size: [1, 512, 30, 32], min: -6.56222, max: 7.764719, mean: 0.025833

        ggml_tensor_t *seg_logit = decode_head.forward(ctx, xlist);
        // tensor [seg_logit] size: [1, 150, 240, 256], min: -49.286793, max: 1.367848, mean: -24.109959

        seg_logit = ggml_interpolate(ctx, seg_logit, 1, H);  
        seg_logit = ggml_interpolate(ctx, seg_logit, 0, W); 
        seg_logit = ggml_softmax(ctx, seg_logit, 2/*dim on C*/);
        // tensor [seg_logit] size: [1, 150, 960, 1024], min: 0.0, max: 1.0, mean: 0.006667

        seg_logit = ggml_argmax_ext(ctx, seg_logit, 2/*dim on C*/);
        // tensor [seg_mask] size: [1, 1, 960, 1024], min: 0.0, max: 25.0, mean: 6.479004
        seg_logit = ggml_clamp(ctx, seg_logit, 0, num_classes);
        return seg_logit;
    }
};

#endif // __SEGFORMER__H__
