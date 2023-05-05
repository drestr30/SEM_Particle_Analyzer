"""
-----------------------------------------------------------------------------------
Description: Mini-Xception for a real time Emotion Recognition
"""

import sys
import torch
import torch.nn as nn
from torchsummary import summary
from torch import Tensor
from typing import Callable, Any, Optional, Tuple, List

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
# def conv_bn_relu(in_channels, out_channels, kernel_size=3, stride=1,
#                  padding=0):
#     return nn.Sequential(
#         nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
#                   stride=stride, padding=padding, bias=False),
#         nn.BatchNorm2d(out_channels),
#         nn.ReLU(inplace=True)
#     )
#
# def SeparableConv2D(in_channels, out_channels, kernel=3):
#     return nn.Sequential(
#         nn.Conv2d(in_channels, in_channels, kernel_size=kernel, stride=1,
#                   groups=in_channels, padding=1, bias=False),
#         nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1,
#                   bias=False)
#     )
#
# class ResidualXceptionBlock(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel=3):
#         super(ResidualXceptionBlock, self).__init__()
#         global device
#
#         self.depthwise_conv1 = SeparableConv2D(in_channels, out_channels,
#                                                kernel).to(device)
#         self.bn1 = nn.BatchNorm2d(out_channels)
#         self.relu1 = nn.ReLU(inplace=True)
#
#         self.depthwise_conv2 = SeparableConv2D(out_channels, out_channels,
#                                                kernel).to(device)
#         self.bn2 = nn.BatchNorm2d(out_channels)
#
#         # self.padd = nn.ZeroPad2d(22)
#         self.maxpool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
#         # self.residual_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2, padding=22, bias=False)
#         self.residual_conv = nn.Conv2d(in_channels, out_channels,
#                                        kernel_size=1, stride=1, padding=0,
#                                        bias=False)
#         self.residual_bn = nn.BatchNorm2d(out_channels)
#
#     def forward(self, x):
#         # residual branch
#         residual = self.residual_conv(x)
#         residual = self.residual_bn(residual)
#
#         # print('input',x.shape)
#         # feature extraction branch
#         x = self.depthwise_conv1(x)
#         # print('conv1',x.shape)
#         x = self.bn1(x)
#         x = self.relu1(x)
#
#         x = self.depthwise_conv2(x)
#         x = self.bn2(x)
#         # print('conv2',x.shape)
#
#         # x = self.padd(x)
#         x = self.maxpool(x)
#         # print(x[:,:, 11:22, 11:22])
#         # print('max_pooling',x.shape)
#         # print('res',residual.shape)
#         return x + residual
#
# class Mini_Xception(nn.Module):
#     def __init__(self, num_clases:int = 7):
#         super(Mini_Xception, self).__init__()
#
#         # self.conv1 = conv_bn_relu(1, 32, kernel_size=3, stride=1, padding=0)
#         # self.conv2 = conv_bn_relu(32, 64, kernel_size=3, stride=1, padding=0)
#         # self.residual_blocks = nn.ModuleList([
#         #     ResidualXceptionBlock(64 , 128).to(device),
#         #     ResidualXceptionBlock(128, 256).to(device),
#         #     ResidualXceptionBlock(256, 512).to(device),
#         #     ResidualXceptionBlock(512, 1024).to(device)
#         # ])
#
#         # self.conv3 = nn.Conv2d(1024, 7, kernel_size=3, stride=1, padding=1)
#
#         self.conv1 = conv_bn_relu(1, 8, kernel_size=3, stride=1, padding=0)
#         self.conv2 = conv_bn_relu(8, 8, kernel_size=3, stride=1, padding=0)
#         self.residual_blocks = nn.ModuleList([
#             ResidualXceptionBlock(8, 16).to(device),
#             ResidualXceptionBlock(16, 32).to(device),
#             ResidualXceptionBlock(32, 64).to(device),
#             ResidualXceptionBlock(64, 128).to(device)
#         ])
#         self.conv3 = nn.Conv2d(128, num_clases, kernel_size=3, stride=1, padding=1)
#
#         self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.softmax = nn.Softmax(dim=1)
#
#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.conv2(x)
#
#         for block in self.residual_blocks:
#             x = block(x)
#             # print('ith block', x.shape, block.device)
#
#         # print('blocks:',x.shape)
#         x = self.conv3(x)
#         # print('conv3',x.shape)
#         x = self.global_avg_pool(x).squeeze(-1).squeeze(-1)
#         x = self.softmax(x)
#         return x

class BasicConv2d(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        activation: str = 'relu',
        **kwargs: Any
    ) -> None:
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.99)
        if activation == 'relu':
            self.activation = nn.ReLU()

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        x = self.bn(x)
        if hasattr(self, 'activation'):
            x = self.activation(x)
        return x

class MiniXception(nn.Module):
    def __init__(self, n_classes, last_activation='softmax'):
        super(MiniXception,self).__init__()

        first_convs = 8
        grow_rate = 2

        self.conv1 = BasicConv2d(1, first_convs, kernel_size=3)
        self.conv2 = BasicConv2d(first_convs, first_convs, kernel_size=3)

        self.deepresmodule1 = DeepwiseResidualModule(first_convs, 16)
        self.deepresmodule2 = DeepwiseResidualModule(16, 32)
        self.deepresmodule3 = DeepwiseResidualModule(32, 64)
        self.deepresmodule4 = DeepwiseResidualModule(64, 128)

        self.classifier = nn.Sequential(nn.Conv2d(128, n_classes, kernel_size=3, padding=1),
                                        nn.AdaptiveAvgPool2d((1, 1)))
        if last_activation is 'sigmoid':
            self.last_activation = nn.Sigmoid()
        elif last_activation is 'softmax':
            self.last_activation = nn.Softmax(dim=-1)


    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.deepresmodule1(x)
        x = self.deepresmodule2(x)
        x = self.deepresmodule3(x)
        x = self.deepresmodule4(x)
        x = self.classifier(x)#.squeeze(-1).squeeze(-1)
        x = x.view((x.shape[0], -1))
        if hasattr(self, 'last_activation'):
            x = self.last_activation(x)
        return x

class DeepwiseResidualModule(nn.Module):
    def __init__(self, in_channels, out_channels) -> None:
        super(DeepwiseResidualModule, self).__init__()
        self.residual = BasicConv2d(in_channels, out_channels,
                                    activation=None, kernel_size=1, stride=2)

        self.separable1 = SeparableConv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.separable2 = SeparableConv2d(out_channels, out_channels, kernel_size=3, padding=1, activation=None)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        res = self.residual(x)
        x = self.separable1(x)
        x = self.separable2(x)
        x = self.pool(x)
        return torch.add(res, x)

class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels,
                 activation: Any = 'relu',
                 kernel_size: int = 1,
                 stride=1,
                 padding=0,
                 dilation=1,
                 bias=False,
                 **kwargs: Any) -> None:
        super(SeparableConv2d, self).__init__()

        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, dilation, groups=in_channels,
                                   bias=bias)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, 1, 0, 1, 1, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels, momentum=0.99, eps=1e-3)

        if activation == 'relu':
            self.activation = nn.ReLU()

        # (2 * (output - 1) - input - kernel) * (1 / stride)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        if hasattr(self, 'activation'):
            x = self.activation(x)
        return x

if __name__ == '__main__':
    import numpy as np
    from torchsummary import summary

    CLASS_NAMES = ['Angry', 'Disgust']

    dummy = torch.ones([1,1,48,48]).cuda()

    model = MiniXception(len(CLASS_NAMES), last_activation='sigmoid').cuda()

    summary(model, input_data=(1, 48, 48))
    y = model(dummy)
    print('y.shape: ', y.shape)
    # model.load_state_dict(torch.load('./saved_models/IOPTFacial_sgd_balanced_checkpoint.pth.tar')['state_dict'])



