from torch import nn

class AlexNet(nn.Module):
    def __init__(self, init_weights=False):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3,
                               out_channels=96, kernel_size=11, stride=4, padding=0)
        # LRN: テンソルのチャンネル方向に対して正規化を行う手法
        # 近いチャンネルの値を使って正規化
        self.local_res_norm1 = nn.modules.normalization.LocalResponseNorm(
            size=1)
        self.max_pool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv2 = nn.Conv2d(in_channels=96,
                               out_channels=256, kernel_size=5, stride=1, padding=1)
        self.local_res_norm2 = nn.modules.normalization.LocalResponseNorm(
            size=1)
        self.max_pool2 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.conv3 = nn.Conv2d(in_channels=256,
                               out_channels=384, kernel_size=3, stride=1, padding=1)

        self.conv4 = nn.Conv2d(in_channels=384,
                               out_channels=384, kernel_size=3, stride=1, padding=1)

        self.conv5 = nn.Conv2d(in_channels=384,
                               out_channels=256, kernel_size=3, stride=1, padding=1)
        self.max_pool3 = nn.MaxPool2d(kernel_size=3, stride=2)

        fc1_in_features = 1024
        # fc1_in_features = 6400
        self.fc1 = nn.Linear(in_features=fc1_in_features, out_features=4096)
        self.fc2 = nn.Linear(in_features=4096, out_features=4096)
        self.fc3 = nn.Linear(in_features=4096, out_features=2)

        if init_weights:
            self._initialize_weights()

    def _initialize_weights(self):
        """
        ref: https://github.com/minar09/VGG16-PyTorch/blob/master/vgg.py#L46-L58
        comment: 重みの初期化
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, inputs):
        """
        Args:
            input(torch.tensor): 227x227x3
        """
        outputs = inputs
        outputs = self.conv1(outputs)
        outputs = nn.functional.relu(outputs)
        # output = nn.ReLU()(output)
        outputs = self.local_res_norm1(outputs)
        outputs = self.max_pool1(outputs)

        outputs = self.conv2(outputs)
        outputs = nn.functional.relu(outputs)
        outputs = self.local_res_norm2(outputs)
        outputs = self.max_pool2(outputs)

        outputs = self.conv3(outputs)
        outputs = nn.functional.relu(outputs)

        outputs = self.conv4(outputs)
        outputs = nn.functional.relu(outputs)

        outputs = self.conv5(outputs)
        outputs = nn.functional.relu(outputs)
        outputs = self.max_pool3(outputs)

        # print(outputs.shape)

        # ↓ バッチ中の各データ単位で flatten する。
        # 全結合層の前にこれを入れないとバッチ処理できない。
        outputs = outputs.view(outputs.size(0), -1)
        # outputs = torch.flatten(outputs)
        outputs = nn.Dropout(p=0.5)(outputs)
        outputs = self.fc1(outputs)
        outputs = nn.functional.relu(outputs)
        outputs = nn.Dropout(p=0.5)(outputs)
        outputs = self.fc2(outputs)
        outputs = nn.functional.relu(outputs)
        outputs = self.fc3(outputs)
        outputs = nn.functional.relu(outputs)

        outputs = nn.Softmax(dim=1)(outputs)

        return outputs
