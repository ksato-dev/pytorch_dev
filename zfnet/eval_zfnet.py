import sys
sys.path.append("../")
from my_utils import HymenopteraDataSet, MyTransfroms
from model import ZFNet
import torch
import tqdm

if __name__ == "__main__":
    val_img_dir_path = "/home/ksato-thinkbook/dataset/hymenoptera_data/val/"

    # resize = 227
    # resize = 128
    resize = 224
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    transforms = MyTransfroms(resize, mean, std)

    val_dataset = HymenopteraDataSet(val_img_dir_path, transforms)
    input, label = val_dataset[0]
    print(input.shape)
    print(label)

    test_data_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=1, shuffle=True)

    print(test_data_loader)

    net = ZFNet()
    net.load_state_dict(torch.load('model_weights.pth'))
    net.eval()

    sizeof_dataset = len(test_data_loader.dataset)
    print("size of dataset:", sizeof_dataset)

    corrects = 0

    for inputs, labels in test_data_loader:
        # train ---
        outputs = net(inputs)
        _, preds = torch.max(outputs, 1)  # 一番スコアが高い ID のリスト
        corrects += torch.sum(preds == labels.data).item()  # 正答率

        print("match:", preds == labels.data)

        # --- train
    print("corrects:", corrects / sizeof_dataset)
    print("========================")

