import sys

sys.path.append("../")

from my_utils import HymenopteraDataSet, MyTransfroms
# from model import VGG16
import torchvision
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

    # network = VGG16(init_weights=True)
    network = torchvision.models.vgg16(pretrained=True)
    network.train()

    network.classifier[6] = torch.nn.Linear(in_features=4096, out_features=2)

    params_to_update = ["classifier.6.weight", "classifier.6.bias"]
    for name, params in network.named_parameters():
        if name in params_to_update:
            params.requires_grad = True
        else:
            params.requires_grad = False
    
    opt = torch.optim.SGD(network.parameters(), lr=1e-3, momentum=0.9)
    loss_func = torch.nn.CrossEntropyLoss()

    num_epochs = 10

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

    torch.save(network.state_dict(), "transfered_model.pth")


if __name__ == "__main__":
    main()
