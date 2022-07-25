import sys
sys.path.append("../")

from my_utils import HymenopteraDataSet, MyTransfroms
import torch
from model import VGG16
import torchvision
import time
import random
import numpy as np


def main():
    random.seed(1234)
    np.random.seed(1234)
    torch.manual_seed(1234)
    
    val_img_dir_path = "/home/ksato-thinkbook/dataset/hymenoptera_data/val/"

    resize = 224
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    transforms = MyTransfroms(resize, mean, std)
    val_dataset = HymenopteraDataSet(
        data_dir_path=val_img_dir_path, transform=transforms)

    val_data_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=1, shuffle=True)

    # Original VGG16
    network = VGG16(init_weights=True)
    network.load_state_dict(torch.load("model.pth"))

    # Official VGG16
    # network = torchvision.models.vgg16()
    # network.classifier[6] = torch.nn.Linear(4096, 2)
    # network.load_state_dict(torch.load("transfered_model.pth"))

    network.eval()

    total_corrects = 0
    avg_infer_time = 0
    sizeof_dataset = len(val_dataset)
    print("sizeof_dataset:", sizeof_dataset)
    for imgs, labels in val_data_loader:
        start_time = time.time()
        outputs = network(imgs)
        end_time = time.time()
        period = end_time - start_time
        print("inference time: ", period)
        avg_infer_time += period

        _, pred_labels = torch.max(outputs, 1)

        corrects = torch.sum(pred_labels == labels).item()
        # print(corrects)
        total_corrects += corrects
        
    print("corrects:", total_corrects / sizeof_dataset)
    print("avg_infer_time:", avg_infer_time / sizeof_dataset)

if __name__ == "__main__":
    main()