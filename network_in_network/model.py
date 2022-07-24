import torch


class NetworkInNetwork(torch.nn.Module):
    def __init__(self, class_num=2, init_weights=False):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(
            kernel_size=5, in_channels=3, out_channels=192, stride=1, padding=2)
        self.conv2 = torch.nn.Conv2d(
            kernel_size=1, in_channels=192, out_channels=160, stride=1, padding=0)
        self.conv3 = torch.nn.Conv2d(
            kernel_size=1, in_channels=160, out_channels=96, stride=1, padding=0)
        self.max_pool1 = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=0)

        self.conv4 = torch.nn.Conv2d(
            kernel_size=5, in_channels=96, out_channels=192, stride=1, padding=2)
        self.conv5 = torch.nn.Conv2d(
            kernel_size=1, in_channels=192, out_channels=192, stride=1, padding=0)
        self.conv6 = torch.nn.Conv2d(
            kernel_size=1, in_channels=192, out_channels=192, stride=1, padding=0)
        self.max_pool2 = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=0)

        self.conv7 = torch.nn.Conv2d(
            kernel_size=5, in_channels=192, out_channels=192, stride=1, padding=2)
        self.conv8 = torch.nn.Conv2d(
            kernel_size=1, in_channels=192, out_channels=160, stride=1, padding=0)

        self.class_num = class_num
        self.conv9 = torch.nn.Conv2d(
            kernel_size=1, in_channels=160, out_channels=self.class_num, stride=1, padding=0)
        self.global_avg_pool = torch.nn.AdaptiveAvgPool2d((1, 1))

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
        outputs = torch.nn.functional.relu(self.conv2(outputs))
        outputs = torch.nn.functional.relu(self.conv3(outputs))
        outputs = self.max_pool1(outputs)
        outputs = torch.nn.Dropout2d(p=0.3)(outputs)

        outputs = torch.nn.functional.relu(self.conv4(outputs))
        outputs = torch.nn.functional.relu(self.conv5(outputs))
        outputs = torch.nn.functional.relu(self.conv6(outputs))
        outputs = self.max_pool2(outputs)
        outputs = torch.nn.Dropout2d(p=0.3)(outputs)

        outputs = torch.nn.functional.relu(self.conv7(outputs))
        outputs = torch.nn.functional.relu(self.conv8(outputs))
        outputs = torch.nn.functional.relu(self.conv9(outputs))

        outputs = self.global_avg_pool(outputs)

        outputs = outputs.view((outputs.shape[0], -1))  # バッチ処理に対応させる

        outputs = torch.nn.functional.softmax(outputs, dim=1)

        return outputs
