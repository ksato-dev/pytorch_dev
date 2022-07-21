from PIL import Image
from model import AlexNet
import torchvision
import torch

if __name__ == "__main__":
    torch.manual_seed(1234)

    # resize = 227
    resize = 128
    pil_img = Image.open("data/cat.jpeg")
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize(size=(resize, resize)),
        torchvision.transforms.ToTensor()
    ])

    torch_img = transforms(pil_img)
    print(torch_img.shape)

    net = AlexNet()
    net.eval()
    input = torch_img
    output = net(input)
    # print(net)
    print(output)
