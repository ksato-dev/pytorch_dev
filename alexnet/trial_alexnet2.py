"""Don't move.
"""

from PIL import Image
from my_utils import AlexNet
import torchvision
import torch
import glob
import numpy as np

if __name__ == "__main__":
    torch.manual_seed(1234)
    
    img_path_list = glob.glob(
        "/home/ksato-thinkbook/dataset/hymenoptera_data/train/ants/*.jpg")

    skip_num = 16
    batch_img = list()
    for p_id, path in enumerate(img_path_list):
        if not p_id % skip_num:

            # resize = 227
            resize = 128
            pil_img = Image.open(path)
            transforms = torchvision.transforms.Compose([
                torchvision.transforms.Resize(size=(resize, resize)),
                torchvision.transforms.ToTensor()
            ])

            torch_img = transforms(pil_img)
            input = torch_img.to("cpu")
            batch_img.append(input)

        batch_img = torch.Tensor(batch_img)

        net = AlexNet()
        net.train()
        # exit()
        # print(net)

        # train ---
        loss_func = torch.nn.CrossEntropyLoss()
        opt = torch.optim.SGD(net.parameters(), lr=1e-2)
    
        output = net(batch_img)
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
        batch_img = list()