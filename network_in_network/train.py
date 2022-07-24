import sys

from sklearn.model_selection import PredefinedSplit
sys.path.append("../")

from my_utils import HymenopteraDataSet, MyTransfroms
from model import NetworkInNetwork
import torch
import tqdm

def main():
    torch.manual_seed(1234)
    
    data_dir_path = "/home/ksato-thinkbook/dataset/hymenoptera_data"
    
    resize = 224
    mean = (0.485, 0.256, 0.406)
    std = (0.229, 0.224, 0.225)
    transforms = MyTransfroms(resize, mean, std)
    train_dataset = HymenopteraDataSet(data_dir_path=(
        data_dir_path + "/train"), transform=transforms)

    # print(train_dataset)
    # print(train_dataset[0])

    train_data_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=16, shuffle=True)

    # network
    model = NetworkInNetwork(init_weights=True)
    model.train()

    num_epochs = 100

    loss_func = torch.nn.CrossEntropyLoss()

    # optimizer を初期化する前に、ネットワークの各パラメータの勾配計算を有効にする。
    for params in model.parameters():
        params.requires_grad = True
    optimizer = torch.optim.SGD(model.parameters(), lr = 1e-3, momentum=0.9)

    size_dataset = len(train_dataset)
    print("size of dataset:", size_dataset)

    torch.set_grad_enabled(True)  # 勾配計算を有効にする。これをやらないと学習できない。
    for epoch in range(num_epochs):
        print("training count:", epoch)

        total_loss = 0
        total_corrects = 0
        for imgs, labels in tqdm.tqdm(train_data_loader):
            optimizer.zero_grad()  # バッチ単位で勾配を０にする。
            # print(imgs, labels)
            outputs = model(imgs)
            _, pred_labels = torch.max(outputs, 1)
            # print("outputs:", outputs)
            loss = loss_func(outputs, labels)
            # print("preds_labels:", pred_labels)
            total_loss += loss.item() * imgs.size(0)
            corrects = torch.sum(pred_labels == labels.data).item()
            total_corrects += corrects

            loss.backward()
            optimizer.step()
    
        print("total_loss:", total_loss / size_dataset)
        print("total_corrects:", total_corrects / size_dataset)
        print("=========================")

    torch.save(model.state_dict(), 'model_weights.pth')

if __name__ == "__main__":
    main()
