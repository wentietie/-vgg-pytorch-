from torch import nn

class VGG(nn.Module):
    def __init__(self, num_classes=1000):
        super(VGG, self).__init__()
        layers = []
        in_dim = 3
        out_dim = 64
        # 循环构造卷积层， 一共有13层
        for i in range(13):
            layers += [nn.Conv2d(in_dim, out_dim, 3, 1, 1), nn.ReLU(inplace=True)]
            in_dim = out_dim
            # 在2 4 7 10 13 的卷积层后添加池化层
            if i == 1 or i == 3 or i == 6 or i == 9 or i == 12:
                layers += [nn.MaxPool2d(2, 2)]
                # 第10个卷积通道数不变 其他的增大2倍
                if i != 9:
                    out_dim *= 2
        self.features = nn.Sequential(*layers)
        # 3个全连接层
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
    def forward(self, x) :
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
