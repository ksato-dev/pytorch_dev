from torch import nn
import torch


class AlexNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3,
                               out_channels=96, kernel_size=11, stride=4, padding=0)
        # LRN: テンソルのチャンネル方向に対して正規化を行う手法
        # 近いチャンネルの値を使って正規化
        self.local_res_norm1 = nn.modules.normalization.LocalResponseNorm(size=1)
        self.max_pool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv2 = nn.Conv2d(in_channels=96,
                               out_channels=256, kernel_size=5, stride=1, padding=1)
        self.local_res_norm2 = nn.modules.normalization.LocalResponseNorm(size=1)
        self.max_pool2 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.conv3 = nn.Conv2d(in_channels=256,
                               out_channels=384, kernel_size=3, stride=1, padding=1)

        self.conv4 = nn.Conv2d(in_channels=384,
                               out_channels=384, kernel_size=3, stride=1, padding=1)

        self.conv5 = nn.Conv2d(in_channels=384,
                               out_channels=256, kernel_size=3, stride=1, padding=1)
        self.max_pool3 = nn.MaxPool2d(kernel_size=3, stride=2)

        fc1_in_features = 1024
        self.fc1 = nn.Linear(in_features=fc1_in_features, out_features=4096)
        self.fc2 = nn.Linear(in_features=4096, out_features=4096)
        self.fc3 = nn.Linear(in_features=4096, out_features=2)

    def forward(self, input):
        """
        Args:
            input(torch.tensor): 227x227x3
        """
        output = input
        output = self.conv1(output)
        output = nn.functional.relu(output)
        # output = nn.ReLU()(output)
        output = self.local_res_norm1(output)
        output = self.max_pool1(output)

        output = self.conv2(output)
        output = nn.functional.relu(output)
        output = self.local_res_norm2(output)
        output = self.max_pool2(output)

        output = self.conv3(output)
        output = nn.functional.relu(output)

        output = self.conv4(output)
        output = nn.functional.relu(output)

        output = self.conv5(output)
        output = nn.functional.relu(output)
        output = self.max_pool3(output)

        output = torch.flatten(output)
        output = nn.Dropout(p=0.5)(output)
        output = self.fc1(output)
        output = nn.functional.relu(output)
        output = nn.Dropout(p=0.5)(output)
        output = self.fc2(output)
        output = nn.functional.relu(output)
        output = self.fc3(output)
        output = nn.functional.relu(output)

        return output
