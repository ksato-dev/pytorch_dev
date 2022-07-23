from my_utils import MyDataSet, MyTransfroms
from model import AlexNet
import torch
import tqdm

if __name__ == "__main__":
    torch.manual_seed(1234)
    
    train_img_dir_path = "/home/ksato-thinkbook/dataset/hymenoptera_data/train/"

    # resize = 227
    resize = 128
    # resize = 224
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    transforms = MyTransfroms(resize, mean, std)

    train_dataset = MyDataSet(train_img_dir_path, transforms)
    input, label = train_dataset[0]
    print(input.shape)
    print(label)

    train_data_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=16, shuffle=True)

    print(train_data_loader)

    net = AlexNet(init_weights=True)
    net.train()
    loss_func = torch.nn.CrossEntropyLoss()

    for param in net.parameters():
        param.requires_grad = True

    # parameters の requires_grad を更新する前に初期化すると終わる。
    opt = torch.optim.SGD(net.parameters(), lr=1e-3, momentum=0.9)

    num_epochs = 100
    sizeof_dataset = len(train_data_loader.dataset)
    torch.set_grad_enabled(True)
    for epoch in tqdm.tqdm(range(num_epochs)):
        total_loss = 0
        corrects = 0

        for inputs, labels in train_data_loader:
            # train ---
            opt.zero_grad()
            outputs = net(inputs)
            _, preds = torch.max(outputs, 1)  # 一番スコアが高い ID のリスト
            loss = loss_func(outputs, labels)
            # print("loss:", loss)
            total_loss += loss.item() * inputs.size(0)  
            corrects += torch.sum(preds == labels.data).item()  # 正答率

            loss.backward()
            opt.step()

            # --- train
        print("epoch:", epoch)
        print("total loss:", total_loss / sizeof_dataset)
        print("corrects:", corrects / sizeof_dataset)
        print("========================")
