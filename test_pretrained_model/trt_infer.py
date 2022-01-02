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

mode = "fp16"
# mode = "int8"

input_img_path = "/opt/dataset/pytorch_advanced/1_image_classification/data/goldenretriever-3724972_640.jpg"
# input_img_path = "/opt/dataset/pytorch_advanced/1_image_classification/data/yellow_car.jpg"


def create_ranking(out):
    """create ranking"""
    ILSVRC_class_index = json.load(
        open(
            "/opt/dataset/pytorch_advanced/1_image_classification/data/imagenet_class_index.json",
            "r",
        )
    )

    length_out = len(out[0])
    ranking = []
    for label_id in range(0, length_out):
        label_name = ILSVRC_class_index[str(label_id)][1]
        ranking.append((label_name, out[0][label_id]))

    ranking.sort(key=lambda x: x[1], reverse=True)  # index 1 means second element

    # print(ranking)
    limit_rank = 10
    for rank, top_label_info in enumerate(ranking):
        print(rank + 1, top_label_info[0])
        if rank >= limit_rank:
            break


if __name__ == "__main__":
    pil_img = Image.open(input_img_path)

    trans_pipeline = tv_transforms.Compose(
        [
            tv_transforms.Resize(resize),
            tv_transforms.CenterCrop(resize),
            tv_transforms.ToTensor(),
            tv_transforms.Normalize(mean, std),
        ]
    )

    img_transformed = trans_pipeline(pil_img)

    batch_input = img_transformed.unsqueeze(0)
    batch_input = batch_input.cuda()

    trt_model = torch2trt.TRTModule()
    if mode == "fp16":
        trt_model.load_state_dict(torch.load("efficientnet_b4.fp16.pth"))
    elif mode == "int8":
        trt_model.load_state_dict(torch.load("efficientnet_b4.int8.pth"))

    start_time = time.time()
    out = trt_model(batch_input)
    end_time = time.time()
    print(end_time - start_time, "[sec]")

    # print(img_transformed)
    # print(out)

    np_out = out.detach().cpu().numpy()

    max_id = np.argmax(np_out)
    print(max_id)

    create_ranking(out)

    del batch_input
    del out
