import torch
import torchvision
from torchvision import models as tv_models
from torchvision import transforms as tv_transforms
import numpy as np
import json
from PIL import Image
import time
import torch2trt

# from torchvision.models.inception import BasicConv2d

resize = 224
# resize = 1024
mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)

mode = "int8"
# mode = "fp16"


if __name__ == "__main__":
    pil_img = Image.open(
        "/opt/dataset/pytorch_advanced/1_image_classification/data/"
        "goldenretriever-3724972_640.jpg"
    )
    # pil_img = Image.open(
    #     "/opt/dataset/pytorch_advanced/1_image_classification/data/yellow_car.jpg"
    # )

    trans_pipeline = tv_transforms.Compose(
        [
            tv_transforms.Resize(resize),
            tv_transforms.CenterCrop(resize),
            tv_transforms.ToTensor(),
            tv_transforms.Normalize(mean, std),
        ]
    )

    img_transformed = trans_pipeline(pil_img)
    # model = tv_models.vgg16(pretrained=True)
    model = tv_models.efficientnet_b4(pretrained=True)
    model.eval()
    model.cuda()

    batch_input = img_transformed.unsqueeze(0)
    batch_input = batch_input.cuda()

    x = torch.ones(batch_input.size()).cuda()  # dummy-data for trt

    if mode == "fp16":
        trt_model = torch2trt.torch2trt(model, [x], fp16_mode=True)
        torch.save(trt_model.state_dict(), "efficientnet_b4.fp16.pth")
    elif mode == "int8":
        trt_model = torch2trt.torch2trt(model, [x], int8_mode=True)
        torch.save(trt_model.state_dict(), "efficientnet_b4.int8.pth")

    del batch_input
