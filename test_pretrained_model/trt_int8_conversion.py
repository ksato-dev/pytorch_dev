import torch
import torchvision
from torchvision import models as tv_models
from torchvision import transforms as tv_transforms
from torchvision import datasets as tv_datasets
import numpy as np
import json
from PIL import Image
import time
import torch2trt

resize = 224
# resize = 1024
mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)
input_img_path = "/opt/dataset/pytorch_advanced/1_image_classification/data/goldenretriever-3724972_640.jpg"
# input_img_path = "/opt/dataset/pytorch_advanced/1_image_classification/data/yellow_car.jpg"
calib_imgs_dir_path = (
    "/opt/dataset/pytorch_advanced/1_image_classification/"
    "data/hymenoptera_data/val"
)

trans_pipeline = tv_transforms.Compose(
    [
        tv_transforms.Resize(resize),
        tv_transforms.CenterCrop(resize),
        tv_transforms.ToTensor(),
        tv_transforms.Normalize(mean, std),
    ]
)


class ImageFolderCalibDataset:
    def __init__(self, root, trans_pipeline):
        self.dataset = tv_datasets.ImageFolder(
            root=root,
            transform=trans_pipeline,
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, _ = self.dataset[idx]
        # image = image[None, ...]  # add batch dimension
        return [image]


if __name__ == "__main__":
    pil_img = Image.open(input_img_path)

    img_transformed = trans_pipeline(pil_img)
    # model = tv_models.vgg16(pretrained=True)
    model = tv_models.efficientnet_b4(pretrained=True)
    model.eval()
    model.cuda()

    batch_input = img_transformed.unsqueeze(0)
    batch_input = batch_input.cuda()

    x = torch.ones(batch_input.size()).cuda()  # dummy-data for trt
    dataset = ImageFolderCalibDataset(
        root=calib_imgs_dir_path, trans_pipeline=trans_pipeline
    )

    trt_model = torch2trt.torch2trt(
        model, [x], int8_mode=True, int8_calib_dataset=dataset
    )
    torch.save(trt_model.state_dict(), "efficientnet_b4.int8.pth")

    del batch_input
