from select import epoll
from my_utils import HymenopteraDataSet, MyTransfroms
from model import AlexNet
from vgg import VGG16
import torchvision
import torch
import tqdm

if __name__ == "__main__":
    torch.manual_seed(1234)
    
    train_img_dir_path = "/home/ksato-thinkbook/dataset/hymenoptera_data/train/"

    # resize = 227
    # resize = 128
    resize = 224
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    # pil_img = Image.open(path)
    # transforms = torchvision.transforms.Compose([
    #     torchvision.transforms.Resize(size=(resize, resize)),
    #     torchvision.transforms.ToTensor()
    # ])
    transforms = MyTransfroms(resize, mean, std)

    train_dataset = HymenopteraDataSet(train_img_dir_path, transforms)
    input, label = train_dataset[0]
    print(input.shape)
    print(label)

    train_data_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=16, shuffle=True)

    print(train_data_loader)

    # net = AlexNet(init_weights=True)
    # net = VGG16()
    net = torchvision.models.vgg16(pretrained=True)
    net.train()
    loss_func = torch.nn.CrossEntropyLoss()

    # update_param_names1 = "features"
    # update_param_names2 = [
    #     "classifier.0.weight", "classifier.0.bias",
    #     "classifier.3.weight", "classifier.3.bias",
    #     "classifier.6.weight", "classifier.6.bias"
    #     ]
    update_param_names2 = [
        "classifier.6.weight", "classifier.6.bias"]
    net.classifier[6] = torch.nn.Linear(in_features=4096, out_features=2)
    for name, param in net.named_parameters():
        # if update_param_names1 in name:
        #     param.requires_grad = True
        #     print(name)
        if name in update_param_names2:
            param.requires_grad = True
            print(name)
        else:
            param.requires_grad = False

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
            # print(outputs, labels)
            # outputs = outputs.unsqueeze_(0)
            # outputs = torch.tensor(outputs)
            _, preds = torch.max(outputs, 1)
            # print("preds:", preds)
            loss = loss_func(outputs, labels)
            # print("loss:", loss)
            total_loss += loss.item() * inputs.size(0)  
            corrects += torch.sum(preds == labels.data).item()

            loss.backward()
            opt.step()

            # --- train
            # batch_img = list()
        test_input, test_label = train_data_loader.dataset[0]
        test_output = net(test_input.unsqueeze_(0))
        _, test_preds = torch.max(test_output, 1)
        print()
        print("test_correct:", torch.sum(
            test_preds == test_label).item() / len(preds))
        print("epoch:", epoch)
        print("total loss:", total_loss / sizeof_dataset)
        print("corrects:", corrects / sizeof_dataset)
        print("========================")
