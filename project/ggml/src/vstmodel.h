#ifndef __VSTMODEL__H__
#define __VSTMODEL__H__
#include "ggml_engine.h"
#include "ggml_nn.h"
#include <vector>

#pragma GCC diagnostic ignored "-Wformat-truncation"

/*
 ResidualBlock1(
  (conv): Sequential(
    (0): ReflectionPad2d((1, 1, 1, 1))
    (1): Conv2d(256, 64, kernel_size=(3, 3), stride=(1, 1))
    (2): ReLU()
    (3): ReflectionPad2d((1, 1, 1, 1))
    (4): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1))
    (5): ReLU()
    (6): ReflectionPad2d((1, 1, 1, 1))
    (7): Conv2d(64, 256, kernel_size=(3, 3), stride=(1, 1))
  )
) */

struct ResidualBlock1 {
    int channel;

    // network params
    struct Conv2d conv_1;
    struct Conv2d conv_4;
    struct Conv2d conv_7;

    void create_weight_tensors(struct ggml_context* ctx) {
        conv_1.in_channels = channel;
        conv_1.out_channels = channel/4;
        conv_1.kernel_size = {3, 3};
        // conv_1.stride = { 1, 1 };
        // conv_1.padding = { 0, 0 };
        // conv_1.dilation = { 1, 1 };
        // conv_1.is_depthwise = false;
        // conv_1.has_bias = true;
        conv_1.create_weight_tensors(ctx);

        conv_4.in_channels = channel/4;
        conv_4.out_channels = channel/4;
        conv_4.kernel_size = {3, 3};
        conv_4.create_weight_tensors(ctx);

        conv_7.in_channels = channel/4;
        conv_7.out_channels = channel;
        conv_7.kernel_size = {3, 3};
        conv_7.create_weight_tensors(ctx);
    }

    void setup_weight_names(const char *prefix) {
        char s[GGML_MAX_NAME];

        snprintf(s, sizeof(s), "%s%s", prefix, "conv.1.");
        conv_1.setup_weight_names(s);

        snprintf(s, sizeof(s), "%s%s", prefix, "conv.4.");
        conv_4.setup_weight_names(s);

        snprintf(s, sizeof(s), "%s%s", prefix, "conv.7.");
        conv_7.setup_weight_names(s);
    }

    ggml_tensor_t* conv_forward(struct ggml_context* ctx, ggml_tensor_t* x) {
        //     self.conv = nn.Sequential(
        //         nn.ReflectionPad2d(1),
        //         nn.Conv2d(channel, channel // 4, kernel_size=3, stride=1, padding=0, bias=True),
        //         nn.ReLU(),
        //         nn.ReflectionPad2d(1),
        //         nn.Conv2d(channel // 4, channel // 4, kernel_size=3, padding=0, bias=True),
        //         nn.ReLU(),
        //         nn.ReflectionPad2d(1),
        //         nn.Conv2d(channel // 4, channel, kernel_size=3, padding=0, bias=True),
        //     )
        x = ggml_replication_pad2d(ctx, x, 1, 1, 1, 1);
        x = conv_1.forward(ctx, x);
        x = ggml_relu(ctx, x);

        x = ggml_replication_pad2d(ctx, x, 1, 1, 1, 1);
        x = conv_4.forward(ctx, x);
        x = ggml_relu(ctx, x);

        x = ggml_replication_pad2d(ctx, x, 1, 1, 1, 1);
        x = conv_7.forward(ctx, x);
        return x;
    }

    // def forward(self, x1, x2) -> Tuple[torch.Tensor, torch.Tensor]:
    //     fx = self.conv(x2)
    //     return (x2, fx + x1)

    // def inverse(self, y1, y2) -> Tuple[torch.Tensor, torch.Tensor]:
    //     fx = self.conv(y1)
    //     return (y2 - fx, y1)

    std::vector<ggml_tensor_t *> forward(struct ggml_context* ctx, ggml_tensor_t* x1, ggml_tensor_t* x2) {
        std::vector<ggml_tensor_t *> xlist;

        ggml_tensor_t *fx = conv_forward(ctx, x2);
        fx = ggml_add(ctx, x1, fx);
        xlist.push_back(x2);
        xlist.push_back(fx);
        return xlist;
    }

    std::vector<ggml_tensor_t *> inverse(struct ggml_context* ctx, ggml_tensor_t* y1, ggml_tensor_t* y2) {
        std::vector<ggml_tensor_t *> ylist;

        ggml_tensor_t *fx = conv_forward(ctx, y1);
        fx = ggml_sub(ctx, y2, fx);
        ylist.push_back(fx);
        ylist.push_back(y1);
        return ylist;
    }
};

/*
 ResidualBlock2(
  (conv): Sequential(
    (0): ReflectionPad2d((1, 1, 1, 1))
    (1): Conv2d(64, 64, kernel_size=(3, 3), stride=(2, 2))
    (2): ReLU()
    (3): ReflectionPad2d((1, 1, 1, 1))
    (4): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1))
    (5): ReLU()
    (6): ReflectionPad2d((1, 1, 1, 1))
    (7): Conv2d(64, 256, kernel_size=(3, 3), stride=(1, 1))
  )
) */
struct ResidualBlock2 {
    int channel;
    
    // network params
    struct Conv2d conv_1;
    struct Conv2d conv_4;
    struct Conv2d conv_7;

    void create_weight_tensors(struct ggml_context* ctx) {
        conv_1.in_channels = channel/4;
        conv_1.out_channels = channel/4;
        conv_1.kernel_size = {3, 3};
        conv_1.stride = { 2, 2 };
        // conv_1.padding = { 0, 0 };
        // conv_1.dilation = { 1, 1 };
        // conv_1.is_depthwise = false;
        // conv_1.has_bias = true;
        conv_1.create_weight_tensors(ctx);

        conv_4.in_channels = channel/4;
        conv_4.out_channels = channel/4;
        conv_4.kernel_size = {3, 3};
        conv_4.create_weight_tensors(ctx);

        conv_7.in_channels = channel/4;
        conv_7.out_channels = channel;
        conv_7.kernel_size = {3, 3};
        conv_7.create_weight_tensors(ctx);
    }

    void setup_weight_names(const char *prefix) {
        char s[GGML_MAX_NAME];

        snprintf(s, sizeof(s), "%s%s", prefix, "conv.1.");
        conv_1.setup_weight_names(s);

        snprintf(s, sizeof(s), "%s%s", prefix, "conv.4.");
        conv_4.setup_weight_names(s);

        snprintf(s, sizeof(s), "%s%s", prefix, "conv.7.");
        conv_7.setup_weight_names(s);
    }

    ggml_tensor_t* conv_forward(struct ggml_context* ctx, ggml_tensor_t* x) {
        // self.conv = nn.Sequential(
        //     nn.ReflectionPad2d(1),
        //     nn.Conv2d(channel // 4, channel // 4, kernel_size=3, stride=2, padding=0, bias=True),
        //     nn.ReLU(),
        //     nn.ReflectionPad2d(1),
        //     nn.Conv2d(channel // 4, channel // 4, kernel_size=3, padding=0, bias=True),
        //     nn.ReLU(),
        //     nn.ReflectionPad2d(1),
        //     nn.Conv2d(channel // 4, channel, kernel_size=3, padding=0, bias=True),
        // )

        x = ggml_replication_pad2d(ctx, x, 1, 1, 1, 1);
        x = conv_1.forward(ctx, x);
        x = ggml_relu(ctx, x);

        x = ggml_replication_pad2d(ctx, x, 1, 1, 1, 1);
        x = conv_4.forward(ctx, x);
        x = ggml_relu(ctx, x);

        x = ggml_replication_pad2d(ctx, x, 1, 1, 1, 1);
        x = conv_7.forward(ctx, x);
        return x;
    }

    // def forward(self, x1, x2) -> Tuple[torch.Tensor, torch.Tensor]:
    //     fx = self.conv(x2)
    //     x1 = vstnet_pixel_unshuffle(x1)  # [1, 16, 676, 1200] ==> [1, 64, 338, 600]
    //     x2 = vstnet_pixel_unshuffle(x2)  # [1, 16, 676, 1200] ==> [1, 64, 338, 600]
    //     return (x2, fx + x1)

    // def inverse(self, y1, y2) -> Tuple[torch.Tensor, torch.Tensor]:
    //     y1 = vstnet_pixel_shuffle(y1)
    //     fx = self.conv(y1)
    //     x1 = y2 - fx
    //     x1 = vstnet_pixel_shuffle(x1)
    //     return (x1, y1)

    std::vector<ggml_tensor_t *> forward(struct ggml_context* ctx, ggml_tensor_t* x1, ggml_tensor_t* x2) {
        std::vector<ggml_tensor_t *> xlist;

        ggml_tensor_t *fx = conv_forward(ctx, x2);
        x1 = pixel_nn_unshuffle(ctx, x1, 2);
        x2 = pixel_nn_unshuffle(ctx, x2, 2);
        fx = ggml_add(ctx, fx, x1);

        xlist.push_back(x2);
        xlist.push_back(fx);
        return xlist;
    }

    std::vector<ggml_tensor_t *> inverse(struct ggml_context* ctx, ggml_tensor_t* y1, ggml_tensor_t* y2) {
        std::vector<ggml_tensor_t *> ylist;

        y1 = ggml_shuffle(ctx, y1, 2);
        ggml_tensor_t *fx = conv_forward(ctx, y1);
        ggml_tensor_t *x1 = ggml_sub(ctx, y2, fx);
        x1 = ggml_shuffle(ctx, x1, 2);

        ylist.push_back(x1);
        ylist.push_back(y1);

        return ylist;
    }
};

/*
 ChannelReduction(
  (block_list): ModuleList(
    (0-1): 2 x ResidualBlock1(
      (conv): Sequential(
        (0): ReflectionPad2d((1, 1, 1, 1))
        (1): Conv2d(256, 64, kernel_size=(3, 3), stride=(1, 1))
        (2): ReLU()
        (3): ReflectionPad2d((1, 1, 1, 1))
        (4): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1))
        (5): ReLU()
        (6): ReflectionPad2d((1, 1, 1, 1))
        (7): Conv2d(64, 256, kernel_size=(3, 3), stride=(1, 1))
      )
    )
  )
) */

struct ChannelReduction {
    int sp_steps = 2;

    // network params
    struct ResidualBlock1 block_list_0;
    struct ResidualBlock1 block_list_1;

    void create_weight_tensors(struct ggml_context* ctx) {
        block_list_0.channel = 256;
        block_list_0.create_weight_tensors(ctx);

        block_list_1.channel = 256;
        block_list_1.create_weight_tensors(ctx);
    }

    void setup_weight_names(const char *prefix) {
        char s[GGML_MAX_NAME];
        snprintf(s, sizeof(s), "%s%s", prefix, "block_list.0.");
        block_list_0.setup_weight_names(s);
        snprintf(s, sizeof(s), "%s%s", prefix, "block_list.1.");
        block_list_1.setup_weight_names(s);
    }

    ggml_tensor_t* forward(struct ggml_context* ctx, ggml_tensor_t* x) {
        // x1, x2 = vstnet_split(x)
        // for i, block in enumerate(self.block_list):
        //     (x1, x2) = block(x1, x2)
        // x = vstnet_merge(x1, x2)
        // # spread
        // for _ in range(self.sp_steps):
        //     x = vstnet_pixel_shuffle(x)
        // # tensor [x] size: [1, 32, 576, 1024], min: -0.919658, max: 1.018641, mean: -0.001065
        // return x

        // Split ...
        int n = x->ne[2]/2; // half on channel ...
        ggml_tensor_t *x1 = ggml_nn_slice(ctx, x, 2/*on C*/, 0, n, 1/*step*/);
        ggml_tensor_t *x2 = ggml_nn_slice(ctx, x, 2/*on C*/, n, 2*n, 1/*step*/);
        std::vector<ggml_tensor_t *>xlist;

        // Block forward ...
        xlist = block_list_0.forward(ctx, x1, x2);
        x1 = xlist[0]; x2 = xlist[1];
        xlist = block_list_1.forward(ctx, x1, x2);
        x1 = xlist[0]; x2 = xlist[1];

        // Merge ...
        x = ggml_concat(ctx, x1, x2, 2/*on channel*/);

        // Shuffle ...
        for (int i = 0; i < sp_steps; i++)
            x = ggml_shuffle(ctx, x, 2);

    	return x;
    }

    ggml_tensor_t* inverse(struct ggml_context* ctx, ggml_tensor_t* x) {
        // for _ in range(self.sp_steps):
        //     x = vstnet_pixel_unshuffle(x, size=2)
        // x1, x2 = vstnet_split(x)
        // n = len(self.block_list)  # 2
        // for i in range(n):
        //     for j, block in enumerate(self.block_list):
        //         if j == n - i - 1:
        //             x1, x2 = block(x1, x2)
        // x = vstnet_merge(x1, x2)
        // return x

        // Unshuffle ...
        for (int i = 0; i < sp_steps; i++)
            x = pixel_nn_unshuffle(ctx, x, 2);

        // Split ...
        int n = x->ne[2]/2; // half on channel ...
        ggml_tensor_t *x1 = ggml_nn_slice(ctx, x, 2/*on C*/, 0, n, 1/*step*/);
        ggml_tensor_t *x2 = ggml_nn_slice(ctx, x, 2/*on C*/, n, 2*n, 1/*step*/);

        // Block reverse forward ...
        std::vector<ggml_tensor_t *>xlist = block_list_1.forward(ctx, x1, x2);
        x1 = xlist[0]; x2 = xlist[1];
        xlist = block_list_0.forward(ctx, x1, x2);
        x1 = xlist[0]; x2 = xlist[1];

        // Merge ...
        x = ggml_concat(ctx, x1, x2, 2 /*on C*/);
        return x;
    }
};

/*
 VSTEncoder(
  (inj_pad): InjectivePad(
    (pad): ZeroPad2d((0, 0, 0, 29))
  )
  (stack): ModuleList(
    (0-9): 10 x ResidualBlock1(
      (conv): Sequential(
        (0): ReflectionPad2d((1, 1, 1, 1))
        (1): Conv2d(16, 4, kernel_size=(3, 3), stride=(1, 1))
        (2): ReLU()
        (3): ReflectionPad2d((1, 1, 1, 1))
        (4): Conv2d(4, 4, kernel_size=(3, 3), stride=(1, 1))
        (5): ReLU()
        (6): ReflectionPad2d((1, 1, 1, 1))
        (7): Conv2d(4, 16, kernel_size=(3, 3), stride=(1, 1))
      )
    )
    (10): ResidualBlock2(
      (conv): Sequential(
        (0): ReflectionPad2d((1, 1, 1, 1))
        (1): Conv2d(16, 16, kernel_size=(3, 3), stride=(2, 2))
        (2): ReLU()
        (3): ReflectionPad2d((1, 1, 1, 1))
        (4): Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1))
        (5): ReLU()
        (6): ReflectionPad2d((1, 1, 1, 1))
        (7): Conv2d(16, 64, kernel_size=(3, 3), stride=(1, 1))
      )
    )
    (11-19): 9 x ResidualBlock1(
      (conv): Sequential(
        (0): ReflectionPad2d((1, 1, 1, 1))
        (1): Conv2d(64, 16, kernel_size=(3, 3), stride=(1, 1))
        (2): ReLU()
        (3): ReflectionPad2d((1, 1, 1, 1))
        (4): Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1))
        (5): ReLU()
        (6): ReflectionPad2d((1, 1, 1, 1))
        (7): Conv2d(16, 64, kernel_size=(3, 3), stride=(1, 1))
      )
    )
    (20): ResidualBlock2(
      (conv): Sequential(
        (0): ReflectionPad2d((1, 1, 1, 1))
        (1): Conv2d(64, 64, kernel_size=(3, 3), stride=(2, 2))
        (2): ReLU()
        (3): ReflectionPad2d((1, 1, 1, 1))
        (4): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1))
        (5): ReLU()
        (6): ReflectionPad2d((1, 1, 1, 1))
        (7): Conv2d(64, 256, kernel_size=(3, 3), stride=(1, 1))
      )
    )
    (21-29): 9 x ResidualBlock1(
      (conv): Sequential(
        (0): ReflectionPad2d((1, 1, 1, 1))
        (1): Conv2d(256, 64, kernel_size=(3, 3), stride=(1, 1))
        (2): ReLU()
        (3): ReflectionPad2d((1, 1, 1, 1))
        (4): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1))
        (5): ReLU()
        (6): ReflectionPad2d((1, 1, 1, 1))
        (7): Conv2d(64, 256, kernel_size=(3, 3), stride=(1, 1))
      )
    )
  )
  (channel_reduction): ChannelReduction(
    (block_list): ModuleList(
      (0-1): 2 x ResidualBlock1(
        (conv): Sequential(
          (0): ReflectionPad2d((1, 1, 1, 1))
          (1): Conv2d(256, 64, kernel_size=(3, 3), stride=(1, 1))
          (2): ReLU()
          (3): ReflectionPad2d((1, 1, 1, 1))
          (4): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1))
          (5): ReLU()
          (6): ReflectionPad2d((1, 1, 1, 1))
          (7): Conv2d(64, 256, kernel_size=(3, 3), stride=(1, 1))
        )
      )
    )
  )
) */

struct VSTEncoder : GGMLNetwork {
    // network hparams
    
    struct ResidualBlock1 stack_0_9[10];
    struct ResidualBlock2 stack_10;
    struct ResidualBlock1 stack_11_19[9];
    struct ResidualBlock2 stack_20;
    struct ResidualBlock1 stack_21_29[9];
    
    struct ChannelReduction channel_reduction;

    size_t get_graph_size()
    {
        return GGML_DEFAULT_GRAPH_SIZE * 2; // 2048 * 4
    }

    void create_weight_tensors(struct ggml_context* ctx) {
        // nChannels = [16, 64, 256]

        for (int i = 0; i < 10; i++) {
            stack_0_9[i].channel = 16;
            stack_0_9[i].create_weight_tensors(ctx);
        }

        stack_10.channel = 16;
        stack_10.create_weight_tensors(ctx);

        for (int i = 0; i < 9; i++) {
            stack_11_19[i].channel = 64;
            stack_11_19[i].create_weight_tensors(ctx);
        }

        stack_20.channel = 64;
        stack_20.create_weight_tensors(ctx);

        for (int i = 0; i < 9; i++) {
            stack_21_29[i].channel = 256;
            stack_21_29[i].create_weight_tensors(ctx);
        }

        // use default configuration ...
        channel_reduction.create_weight_tensors(ctx);
    }

    void setup_weight_names(const char *prefix) {
        char s[GGML_MAX_NAME];

        for (int i = 0; i < 10; i++) {
            snprintf(s, sizeof(s), "%sstack.%d.", prefix, i);
            stack_0_9[i].setup_weight_names(s);
        }

        snprintf(s, sizeof(s), "%s%s", prefix, "stack.10.");
        stack_10.setup_weight_names(s);

        for (int i = 0; i < 9; i++) {
            snprintf(s, sizeof(s), "%sstack.%d.", prefix, i + 11);
            stack_11_19[i].setup_weight_names(s);

        }
        snprintf(s, sizeof(s), "%s%s", prefix, "stack.20.");
        stack_20.setup_weight_names(s);

        for (int i = 0; i < 9; i++) {
            snprintf(s, sizeof(s), "%sstack.%d.", prefix, i + 21);
            stack_21_29[i].setup_weight_names(s);

        }

        snprintf(s, sizeof(s), "%s%s", prefix, "channel_reduction.");
        channel_reduction.setup_weight_names(s);
    }

    // ggml_tensor_t* forward(struct ggml_context* ctx, ggml_tensor_t* x) {
    ggml_tensor_t* forward(ggml_context_t* ctx, int argc, ggml_tensor_t* argv[]) {
        GGML_UNUSED(argc);
        ggml_tensor_t* x = argv[0];

        // x = self.inj_pad.forward(x)
        // x1, x2 = vstnet_split(x)
        // for block in self.stack:
        //     x1, x2 = block(x1, x2)
        // x = vstnet_merge(x1, x2)
        // x = self.channel_reduction.forward(x)
        // # tensor [x] size: [1, 32, 676, 1200], min: -1.037655, max: 1.071546, mean: -0.001224
        // return x
        int W = (int)x->ne[0];
        int H = (int)x->ne[1];
        int C = (int)x->ne[2];
        int B = (int)x->ne[3];
        x = ggml_pad(ctx, x, W, H, C+29, B);

        int n = x->ne[2]/2; // half on channel ...
        ggml_tensor_t *x1 = ggml_nn_slice(ctx, x, 2/*on C*/, 0, n, 1/*step*/);
        ggml_tensor_t *x2 = ggml_nn_slice(ctx, x, 2/*on C*/, n, 2*n, 1/*step*/);

        // Blocks forward ...
        std::vector<ggml_tensor_t *> xlist;
        // -----------------------------------------------
        for (int i = 0; i < 10; i++) {
            xlist = stack_0_9[i].forward(ctx, x1, x2);
            x1 = xlist[0]; x2 = xlist[1];
        }
        // -----------------------------------------------
        xlist = stack_10.forward(ctx, x1, x2);
        x1 = xlist[0]; x2 = xlist[1];
        // -----------------------------------------------
        for (int i = 0; i < 9; i++) {
            xlist = stack_11_19[i].forward(ctx, x1, x2);
            x1 = xlist[0]; x2 = xlist[1];
        }
        // -----------------------------------------------
        xlist = stack_20.forward(ctx, x1, x2);
        x1 = xlist[0]; x2 = xlist[1];
        // -----------------------------------------------
        for (int i = 0; i < 9; i++) {
            xlist = stack_21_29[i].forward(ctx, x1, x2);
            x1 = xlist[0]; x2 = xlist[1];
        }
        // -----------------------------------------------

        x = ggml_concat(ctx, x1, x2, 2/*on channels*/);
        x = channel_reduction.forward(ctx, x);

    	return x;
    }
};

struct VSTDecoder : GGMLNetwork {
    // network hparams
    
    struct ResidualBlock1 stack_0_9[10];
    struct ResidualBlock2 stack_10;
    struct ResidualBlock1 stack_11_19[9];
    struct ResidualBlock2 stack_20;
    struct ResidualBlock1 stack_21_29[9];
    
    struct ChannelReduction channel_reduction;

    size_t get_graph_size()
    {
        return GGML_DEFAULT_GRAPH_SIZE * 2; // 2048 * 4
    }

    void create_weight_tensors(struct ggml_context* ctx) {
        // nChannels = [16, 64, 256]

        for (int i = 0; i < 10; i++) {
            stack_0_9[i].channel = 16;
            stack_0_9[i].create_weight_tensors(ctx);
        }

        stack_10.channel = 16;
        stack_10.create_weight_tensors(ctx);

        for (int i = 0; i < 9; i++) {
            stack_11_19[i].channel = 64;
            stack_11_19[i].create_weight_tensors(ctx);
        }

        stack_20.channel = 64;
        stack_20.create_weight_tensors(ctx);

        for (int i = 0; i < 9; i++) {
            stack_21_29[i].channel = 256;
            stack_21_29[i].create_weight_tensors(ctx);
        }

        // use default configuration ...
        channel_reduction.create_weight_tensors(ctx);
    }

    void setup_weight_names(const char *prefix) {
        char s[GGML_MAX_NAME];

        for (int i = 0; i < 10; i++) {
            snprintf(s, sizeof(s), "%sstack.%d.", prefix, i);
            stack_0_9[i].setup_weight_names(s);
        }

        snprintf(s, sizeof(s), "%s%s", prefix, "stack.10.");
        stack_10.setup_weight_names(s);

        for (int i = 0; i < 9; i++) {
            snprintf(s, sizeof(s), "%sstack.%d.", prefix, i + 11);
            stack_11_19[i].setup_weight_names(s);

        }
        snprintf(s, sizeof(s), "%s%s", prefix, "stack.20.");
        stack_20.setup_weight_names(s);

        for (int i = 0; i < 9; i++) {
            snprintf(s, sizeof(s), "%sstack.%d.", prefix, i + 21);
            stack_21_29[i].setup_weight_names(s);

        }

        snprintf(s, sizeof(s), "%s%s", prefix, "channel_reduction.");
        channel_reduction.setup_weight_names(s);
    }

    // ggml_tensor_t* forward(struct ggml_context* ctx, ggml_tensor_t* x) {
    ggml_tensor_t* forward(ggml_context_t* ctx, int argc, ggml_tensor_t* argv[]) {
        GGML_UNUSED(argc);
        ggml_tensor_t* x = argv[0];

        // # tensor [x] size: [1, 32, 676, 1200], min: -1.211427, max: 1.173751, mean: 0.000158
        // x = self.channel_reduction.inverse(x)
        // x1, x2 = vstnet_split(x)
        // for block in self.reverse_stack:
        //     x1, x2 = block.inverse(x1, x2)
        // x = vstnet_merge(x1, x2)
        // x = self.inj_pad.inverse(x)
        // # tensor [x] size: [1, 3, 676, 1200], min: -0.478305, max: 1.713724, mean: 0.482254
        // return x.clamp(0.0, 1.0)

        x = channel_reduction.inverse(ctx, x);
        int n = x->ne[2]/2; // half on channel ...
        ggml_tensor_t *x1 = ggml_nn_slice(ctx, x, 2/*on C*/, 0, n, 1/*step*/);
        ggml_tensor_t *x2 = ggml_nn_slice(ctx, x, 2/*on C*/, n, 2*n, 1/*step*/);

        // Blocks inverse forward ...
        std::vector<ggml_tensor_t *> xlist;
        // -----------------------------------------------
        for (int i = 8; i >= 0; i--) {
            xlist = stack_21_29[i].inverse(ctx, x1, x2);
            x1 = xlist[0]; x2 = xlist[1];
        }
        // -----------------------------------------------
        xlist = stack_20.inverse(ctx, x1, x2);
        x1 = xlist[0]; x2 = xlist[1];
        // -----------------------------------------------
        for (int i = 8; i >= 0; i--) {
            xlist = stack_11_19[i].inverse(ctx, x1, x2);
            x1 = xlist[0]; x2 = xlist[1];
        }
        // -----------------------------------------------
        xlist = stack_10.inverse(ctx, x1, x2);
        x1 = xlist[0]; x2 = xlist[1];
        // -----------------------------------------------
        for (int i = 9; i >= 0; i--) {
            xlist = stack_0_9[i].inverse(ctx, x1, x2);
            x1 = xlist[0]; x2 = xlist[1];
        }
        // -----------------------------------------------

        x = ggml_concat(ctx, x1, x2, 2/*on channels*/);
        int W = (int)x->ne[0];
        int H = (int)x->ne[1];
        int C = (int)x->ne[2];
        int B = (int)x->ne[3];
        x = ggml_nn_slice(ctx, x, 2/*on channels*/, 0, C - 29, 1/*step*/);        

        return x;
    }
};

#endif // __VSTMODEL__H__
