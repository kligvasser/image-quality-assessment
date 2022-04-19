import math
import torch.nn as nn
import torch.nn.functional as F

class SEBlock(nn.Module):
    def __init__(self, in_channels, out_channels=None, ratio=16):
        super(SEBlock, self).__init__()
        self.in_channels = in_channels
        if out_channels is None:
            out_channels = in_channels
        self.ratio = ratio
        self.relu = nn.ReLU(True)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.transform = nn.Sequential(
            nn.Linear(in_channels, in_channels // ratio),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // ratio, out_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        x_avg = self.global_pool(x).flatten(1, -1)
        mask = self.transform(x_avg)
        return x * mask.unsqueeze(-1).unsqueeze(-1)

class ResBasicBlock(nn.Module):
    def __init__(self, in_planes, planes,  stride=1, expansion=1, downsample=None, groups=1, residual_block=None):
        super(ResBasicBlock, self).__init__()       
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, groups=groups, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, expansion * planes, kernel_size=3, stride=1, padding=1, groups=groups, bias=False)
        self.bn2 = nn.BatchNorm2d(expansion * planes)
        self.downsample = downsample
        self.residual_block = residual_block
        self.stride = stride
        self.expansion = expansion

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(residual)

        if self.residual_block is not None:
            residual = self.residual_block(residual)

        out += residual
        out = self.relu(out)

        return out

class DSDRegressor(nn.Module):
    def __init__(self, in_channels=3, in_planes=32, width=[32, 32, 64, 64, 128, 128], layers=[2, 2, 2, 2, 2, 2], block=ResBasicBlock, expansion=1, residual_block=SEBlock, groups=[1, 1, 1, 1, 1, 1], num_classes=4):
        super(DSDRegressor, self).__init__()
        # parameters
        self.in_planes = in_planes

        # image to features
        self.image_to_features = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=self.in_planes, kernel_size=5, stride=1, padding=2, bias=False),
            nn.BatchNorm2d(num_features=self.in_planes),
            nn.ReLU(inplace=True))

        # features
        blocks = []
        for i in range(len(layers)):
            blocks.append(self._make_layer(block=block, planes=width[i], blocks=layers[i], expansion=expansion, stride=1 if i == 0 else 2, groups=groups[i], residual_block=residual_block))
        self.features = nn.Sequential(*blocks)

        # pooling
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=1)

        # classifier
        self.classifier = nn.Sequential(
            nn.Linear(width[-1] * expansion, in_planes),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Linear(in_planes, num_classes))

    def _make_layer(self, block, planes, blocks, expansion=1, stride=1, groups=1, residual_block=None):
        downsample = None
        out_planes = planes * expansion
        if stride != 1 or self.in_planes != out_planes:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_planes, out_planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * expansion),
            )
        if residual_block is not None:
            residual_block = residual_block(out_planes)

        layers = []
        layers.append(block(self.in_planes, planes, stride, expansion=expansion, downsample=downsample, groups=groups, residual_block=residual_block))
        self.in_planes = planes * expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_planes, planes, expansion=expansion, groups=groups, residual_block=residual_block))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.image_to_features(x)
        x = self.features(x)
        x = self.avg_pool(x)
        x = x.flatten(start_dim=1)
        x = self.classifier(x)
        return x