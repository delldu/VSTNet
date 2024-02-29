"""Image/Video Photo Style Package."""  # coding=utf-8
#
# /************************************************************************************
# ***
# ***    Copyright Dell 2022(18588220928@163.com) All Rights Reserved.
# ***
# ***    File Author: Dell, 2022年 09月 08日 星期四 01:39:22 CST
# ***
# ************************************************************************************/
#

__version__ = "1.0.0"

import os
from tqdm import tqdm
import torch
import torch.nn.functional as F
from .vstnet import create_photo_style_model, create_artist_style_model

import todos

import pdb


def get_trace_model():
    """Create model."""

    model = create_photo_style_model()

    device = todos.model.get_device()
    model = model.to(device)
    model.eval()

    print(f"Running on {device} ...")
    return model, device

def get_photo_style_model():
    """Create model."""

    model = create_photo_style_model()

    device = todos.model.get_device()
    model = model.to(device)
    model.eval()

    print(f"Running on {device} ...")
    # make sure model good for C/C++
    model = torch.jit.script(model)
    # https://github.com/pytorch/pytorch/issues/52286
    torch._C._jit_set_profiling_executor(False)
    # C++ Reference
    # torch::jit::getProfilingMode() = false;                                                                                                             
    # torch::jit::setTensorExprFuserEnabled(false);

    todos.data.mkdir("output")
    if not os.path.exists("output/image_photo_style.torch"):
        model.save("output/image_photo_style.torch")

    return model, device


def image_photo_predict(input_files, style_file, output_dir):
    # Create directory to store result
    todos.data.mkdir(output_dir)

    # load model
    model, device = get_photo_style_model()

    # load files
    image_filenames = todos.data.load_files(input_files)
    style_tensor = todos.data.load_tensor(style_file)

    # start predict
    progress_bar = tqdm(total=len(image_filenames))
    for filename in image_filenames:
        progress_bar.update(1)

        # orig input
        content_tensor = todos.data.load_tensor(filename)
        B, C, H, W = content_tensor.size()

        with torch.no_grad():
            predict_tensor = model(content_tensor.to(device), style_tensor.to(device))

        output_file = f"{output_dir}/{os.path.basename(filename)}"

        SB, SC, SH, SW = style_tensor.shape
        if SH != H or SW != W:
            style_tensor = F.interpolate(style_tensor, size=(H, W), mode="bilinear", align_corners=False)
        todos.data.save_tensor([content_tensor, style_tensor, predict_tensor], output_file)

    todos.model.reset_device()
