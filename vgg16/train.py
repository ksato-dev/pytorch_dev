from ctypes import sizeof
import sys

sys.path.append("../")

from my_utils import HymenopteraDataSet, MyTransfroms
from model import VGG16
import torch
import tqdm

def main():
    torch.manual_seed(1234)
    train_img_dir_path = "/home/ksato-thinkbook/dataset/hymenoptera_data/train/"

    resize = 224
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    transforms = MyTransfroms(resize, mean, std)
    train_dataset = HymenopteraDataSet(
        data_dir_path=train_img_dir_path, transform=transforms)

    train_data_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=16, shuffle=True)

    sample_imgs, labels = next(iter(train_data_loader))
    # print(data)

    network = VGG16(init_weights=True)
    network.train()
    # sample_output = network(sample_imgs)
    # print(sample_output)

    for params in network.parameters():
        params.requires_grad = True
    
    opt = torch.optim.SGD(network.parameters(), lr=1e-3, momentum=0.9)
    loss_func = torch.nn.CrossEntropyLoss()

    num_epochs = 100

    sizeof_dataset = len(train_dataset)
    print("sizeof_dataset:", sizeof_dataset)

    torch.set_grad_enabled(True)
    for epoch in range(num_epochs):
        print("Current epoch:", epoch)
        total_loss = 0
        total_corrects = 0

        for inputs, labels in tqdm.tqdm(train_data_loader):
            opt.zero_grad()

            outputs = network(inputs)
            # print(outputs)
            _, pred_labels = torch.max(outputs, 1)

            loss = loss_func(outputs, labels)
            loss.backward()
            opt.step()

            total_loss += loss.item() * inputs.size(0)
            total_corrects += torch.sum(pred_labels
                                        == labels).item()
            
        print("total_loss:", total_loss / sizeof_dataset)
        print("total_corrects:", total_corrects / sizeof_dataset)

    torch.save(network.state_dict(), "model.pth")


if __name__ == "__main__":
    main()
