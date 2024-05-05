
import torch.nn as nn
import math
import numpy as np

class residual(nn.Module):
    def __init__(self, input4, output4, stride2, expand_ratio):
        super(residual, self).__init__()
        assert stride2 in [1, 2]
        self.stride2 = stride2
        hDim = int(input4 * expand_ratio)
        self.use_res_connect = self.stride2 == 1 and input4 == output4

        if expand_ratio == 1:
            self.conv = nn.Sequential(
                nn.Conv2d(hDim, hDim, 3, self.stride2, 1, groups=hDim, bias=False),
                nn.BatchNorm2d(hDim),
                nn.ReLU6(inplace=True),
                nn.Conv2d(hDim, output4, 1, 1, 0, bias=False),
                nn.BatchNorm2d(output4),
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(input4, hDim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hDim),
                nn.ReLU6(inplace=True),
                nn.Conv2d(hDim, hDim, 3, self.stride2, 1, groups=hDim, bias=False),
                nn.BatchNorm2d(hDim),
                nn.ReLU6(inplace=True),
                nn.Conv2d(hDim, output4, 1, 1, 0, bias=False),
                nn.BatchNorm2d(output4),
            )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


def divs(input3, div=8):
    value1 = int(np.ceil(input3 * 1. / div) * div)
    return value1


class MobileNetV2(nn.Module):
    def __init__(self, nClass, input_size=224, width_mult=1.):
        super(MobileNetV2, self).__init__()
        block = residual
        self.first_channel = 32
        self.last_channel = 1280
        interverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        assert input_size % 32 == 0
        self.last_channel = divs(self.last_channel * width_mult) if width_mult > 1.0 else self.last_channel

        self.features = [nn.Sequential(
            nn.Conv2d(3, self.first_channel, 3, 2, 1, bias=False),
            nn.BatchNorm2d(self.first_channel),
            nn.ReLU6(inplace=True))]

        for t, c, n, s in interverted_residual_setting:
            output_channel = divs(c * width_mult) if t > 1 else c
            for i in range(n):
                if i == 0:
                    self.features.append(block(self.first_channel, output_channel, s, expand_ratio=t))
                else:
                    self.features.append(block(self.first_channel, output_channel, 1, expand_ratio=t))
                self.first_channel = output_channel

        self.features.append(nn.Sequential(
            nn.Conv2d(self.first_channel, self.last_channel, 1, 1, 0, bias=False),
            nn.BatchNorm2d(self.last_channel),
            nn.ReLU6(inplace=True)
        ))

        self.features = nn.Sequential(*self.features)
        self.classifier = nn.Linear(self.last_channel, nClass)

        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.mean(3).mean(2)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
