from PIL import Image
from model import AlexNet
import torchvision
import torch

if __name__ == "__main__":
    torch.manual_seed(1234)

    # resize = 227
    resize = 128
    pil_img = Image.open("data/0013035.jpg")
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize(size=(resize, resize)),
        torchvision.transforms.ToTensor()
    ])

    torch_img = transforms(pil_img)

    net = AlexNet()
    net.train()
    input = torch_img.to("cpu")
    # exit()
    # print(net)

    # train ---
    loss_func = torch.nn.CrossEntropyLoss()
    opt = torch.optim.SGD(net.parameters(), lr=1e-1)
    
    for i in range(10):
        output = net(input)
        opt.zero_grad()
        output = output.unsqueeze_(0)
        output = torch.tensor(output, requires_grad = True)
        gt_vec = torch.tensor([0])
        max_argid = torch.argmax(output)
        max_argid = torch.tensor([max_argid])
        print("max_id:", max_argid)
        print(output, gt_vec)
        loss = loss_func(output, gt_vec)
        print("loss:", loss)

        loss.backward()
        opt.step()

    # --- train