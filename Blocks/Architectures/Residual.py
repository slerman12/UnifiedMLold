from torch import nn

import Utils


class Residual(nn.Module):
    def __init__(self, module, down_sample=None):
        super().__init__()
        self.module = module
        self.down_sample = down_sample

    def forward(self, x):
        y = self.module(x)
        if self.down_sample is not None:
            x = self.down_sample(x)
        return y + x

    def output_shape(self, height, width):
        return Utils.cnn_output_shape(height, width, self.module)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, down_sample=None):
        super().__init__()

        pre_residual = nn.Sequential(nn.Conv2d(in_channels, out_channels,
                                               kernel_size=3, stride=stride,
                                               padding=1, bias=False),
                                     nn.BatchNorm2d(out_channels),
                                     nn.Conv2d(in_channels, out_channels,
                                               kernel_size=3, padding=1, bias=False),
                                     nn.BatchNorm2d(out_channels))

        self.Residual_block = nn.Sequential(Residual(pre_residual, down_sample), nn.ReLU())

    def forward(self, x):
        return self.Residual_block(x)

    def output_shape(self, height, width):
        return Utils.cnn_output_shape(height, width, self.Residual_block)


