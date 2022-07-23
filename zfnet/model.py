import torch

class ZFNet(torch.nn.Module):
    def __init__(self, init_weights=False):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(kernel_size=7, in_channels=3, out_channels=96, stride=2, padding=0)
        self.max_pool1 = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=0)
        self.conv2 = torch.nn.Conv2d(kernel_size=5, in_channels=96, out_channels=256, stride=2, padding=1)
        self.max_pool2 = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=0)

        self.conv3 = torch.nn.Conv2d(kernel_size=3, in_channels=256, out_channels=384, stride=1, padding=1)
        self.conv4 = torch.nn.Conv2d(kernel_size=3, in_channels=384, out_channels=384, stride=1, padding=1)
        self.conv5 = torch.nn.Conv2d(kernel_size=3, in_channels=384, out_channels=256, stride=1, padding=1)
        self.max_pool3 = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=0)

        self.fc1 = torch.nn.Linear(in_features=6400, out_features=4096)
        self.fc2 = torch.nn.Linear(in_features=4096, out_features=4096)
        self.fc3 = torch.nn.Linear(in_features=4096, out_features=2)

        if init_weights:
            self._initialize_weights()

    def _initialize_weights(self):
        """
        ref: https://github.com/minar09/VGG16-PyTorch/blob/master/vgg.py#L46-L58
        comment: 重みの初期化
        """
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, torch.nn.BatchNorm2d):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, torch.nn.Linear):
                torch.nn.init.normal_(m.weight, 0, 0.01)
                torch.nn.init.constant_(m.bias, 0)

    def forward(self, inputs):
        outputs = inputs
        outputs = torch.nn.functional.relu(self.conv1(outputs))
        outputs = torch.nn.LocalResponseNorm(size=1)(outputs)
        outputs = self.max_pool1(outputs)

        outputs = torch.nn.functional.relu(self.conv2(outputs))
        outputs = torch.nn.LocalResponseNorm(size=1)(outputs)
        outputs = self.max_pool2(outputs)

        outputs = torch.nn.functional.relu(self.conv3(outputs))
        outputs = torch.nn.functional.relu(self.conv4(outputs))
        outputs = torch.nn.functional.relu(self.conv5(outputs))
        outputs = self.max_pool3(outputs)

        outputs = outputs.view(inputs.shape[0], -1)
        outputs = self.fc1(outputs)
        outputs = torch.nn.functional.relu(outputs)
        outputs = self.fc2(outputs)
        outputs = torch.nn.functional.relu(outputs)
        outputs = self.fc3(outputs)
        
        return outputs
