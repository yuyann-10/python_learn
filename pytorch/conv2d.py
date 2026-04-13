# -*- coding: utf-8 -*-
# 作者：小土堆
# 公众号：土堆碎念
import torch
import torchvision
from torch import nn
from torch.nn import Conv2d
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10("../data", train=False, transform=torchvision.transforms.ToTensor(),
                                       download=True)
dataloader = DataLoader(dataset, batch_size=64)

class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        self.conv1 = Conv2d(in_channels=3, out_channels=9, kernel_size=3, stride=1, padding=0)
        #输入三通道，彩色，一个卷积核是3*3*3,一个卷积后输出的是一个通道，如果想要输出6通道，需要6个卷积核

    def forward(self, x):
        x = self.conv1(x)
        return x

tudui = Tudui()
print(tudui)

writer = SummaryWriter("../logs")

step = 0
for data in dataloader:
    imgs, targets = data
    output = tudui(imgs)
    print(imgs.shape)    # input shape: torch.Size([64, 3, 32, 32])
    print(output.shape)  # output shape: torch.Size([64, 6, 30, 30])

    writer.add_images("input", imgs, step)
    # torch.Size([64, 6, 30, 30])  -> [xxx, 3, 30, 30]

    output = torch.reshape(output, (-1, 3, 30, 30))#-1:根据数据形状自动调整批次，输出的是三通道以便可视化
    writer.add_images("output", output, step)

    step = step + 1

writer.close()

