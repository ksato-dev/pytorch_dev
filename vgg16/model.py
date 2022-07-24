import torch


class VGG16(torch.nn.Module):
    def __init__(self, init_weights):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(
            in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(64)
        self.conv2 = torch.nn.Conv2d(
            in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.max_pool1 = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.bn2 = torch.nn.BatchNorm2d(64)

        self.conv3 = torch.nn.Conv2d(
            in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.bn3 = torch.nn.BatchNorm2d(128)
        self.conv4 = torch.nn.Conv2d(
            in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.bn4 = torch.nn.BatchNorm2d(128)
        self.max_pool2 = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.conv5 = torch.nn.Conv2d(
            in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.bn5 = torch.nn.BatchNorm2d(256)
        self.conv6 = torch.nn.Conv2d(
            in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.bn6 = torch.nn.BatchNorm2d(256)
        self.conv7 = torch.nn.Conv2d(
            in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.bn7 = torch.nn.BatchNorm2d(256)
        self.max_pool3 = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.conv8 = torch.nn.Conv2d(
            in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.bn8 = torch.nn.BatchNorm2d(512)
        self.conv9 = torch.nn.Conv2d(
            in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.bn9 = torch.nn.BatchNorm2d(512)
        self.conv10 = torch.nn.Conv2d(
            in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.bn10 = torch.nn.BatchNorm2d(512)
        self.max_pool4 = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.conv11 = torch.nn.Conv2d(
            in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.bn11 = torch.nn.BatchNorm2d(512)
        self.conv12 = torch.nn.Conv2d(
            in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.bn12 = torch.nn.BatchNorm2d(512)
        self.conv13 = torch.nn.Conv2d(
            in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.bn13 = torch.nn.BatchNorm2d(512)
        self.max_pool5 = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        input_size = 25088
        self.fc1 = torch.nn.Linear(in_features=input_size, out_features=256)
        self.fc2 = torch.nn.Linear(in_features=256, out_features=256)

        class_num = 2
        self.fc3 = torch.nn.Linear(in_features=256, out_features=class_num)

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
        outputs = self.bn1(torch.nn.functional.relu(self.conv1(outputs)))
        outputs = self.bn2(torch.nn.functional.relu(self.conv2(outputs)))
        outputs = self.max_pool1(outputs)

        outputs = self.bn3(torch.nn.functional.relu(self.conv3(outputs)))
        outputs = self.bn4(torch.nn.functional.relu(self.conv4(outputs)))
        outputs = self.max_pool2(outputs)

        outputs = self.bn5(torch.nn.functional.relu(self.conv5(outputs)))
        outputs = self.bn6(torch.nn.functional.relu(self.conv6(outputs)))
        outputs = self.bn7(torch.nn.functional.relu(self.conv7(outputs)))
        outputs = self.max_pool3(outputs)

        outputs = self.bn8(torch.nn.functional.relu(self.conv8(outputs)))
        outputs = self.bn9(torch.nn.functional.relu(self.conv9(outputs)))
        outputs = self.bn10(torch.nn.functional.relu(self.conv10(outputs)))
        outputs = self.max_pool4(outputs)

        outputs = self.bn11(torch.nn.functional.relu(self.conv11(outputs)))
        outputs = self.bn12(torch.nn.functional.relu(self.conv12(outputs)))
        outputs = self.bn13(torch.nn.functional.relu(self.conv13(outputs)))
        outputs = self.max_pool5(outputs)

        outputs = outputs.view(inputs.size(0), -1)
        outputs = torch.nn.functional.relu(self.fc1(outputs))
        outputs = torch.nn.Dropout(p=0.5)(outputs)
        outputs = torch.nn.functional.relu(self.fc2(outputs))
        outputs = torch.nn.Dropout(p=0.5)(outputs)

        outputs = torch.nn.functional.relu(self.fc3(outputs))
        outputs = torch.nn.functional.softmax(outputs, dim=1)
        # print(outputs)

        return outputs
