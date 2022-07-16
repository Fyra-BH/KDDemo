from torch import nn
from torch.nn import functional as F

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_blocks=2, downsampling=False):
        super(ResidualBlock, self).__init__()
        self.downsampling = downsampling
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2 if downsampling else 1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            *[nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False) for _ in range(num_blocks-1)],
            nn.BatchNorm2d(out_channels),
        )
        if downsampling:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut= lambda x: x
        
    def forward(self, x):
        out = self.layers(x)
        x = self.shortcut(x)
        return F.relu(out + x)

class ResNet18(nn.Module):
    def __init__(self, in_channels=3, out_features=1000):
        super(ResNet18, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, 64, 7, 2, 3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 1),
            ResidualBlock(64, 64),
            ResidualBlock(64, 64),
            ResidualBlock(64, 128, downsampling=True),
            ResidualBlock(128, 128),
            ResidualBlock(128, 256, downsampling=True),
            ResidualBlock(256, 256),
            ResidualBlock(256, 512, downsampling=True),
            ResidualBlock(512, 512),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Linear(512, out_features)

    def forward(self, x):
        x = self.layers(x)
        x = self.fc(x.view(x.shape[0], -1))
        return x

# 知识蒸馏中的 Student Net
class StudentNet(nn.Module):
    def __init__(self, in_channels=3, out_features=1000):
        super(StudentNet, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, 64, 7, 2, 3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 1),
            ResidualBlock(64, 64),
            ResidualBlock(64, 128, downsampling=True),
            ResidualBlock(128, 128, downsampling=True),
            ResidualBlock(128, 256, downsampling=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Linear(256, out_features)

    def forward(self, x):
        x = self.layers(x)
        x = self.fc(x.view(x.shape[0], -1))
        return x
