import torch
import torch.nn as nn
from torchvision.models import resnet18
import torch.nn.functional as F


class Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True, dilation=1):
        super(Conv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias,
                              dilation=dilation)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class CRNN(nn.Module):
    def __init__(self, vocab_size):
        super(CRNN, self).__init__()
        resnet = resnet18(pretrained=True)
        modules = list(resnet.children())[:-3]
        self.resnet = nn.Sequential(*modules)
        self.fc1 = nn.Linear(1024, 256)
        self.fc2 = nn.Linear(256, vocab_size)
        self.gru1 = nn.GRU(input_size=256, hidden_size=256)

    def forward(self, x):
        x = self.resnet(x)
        x = x.permute(0, 3, 1, 2).contiguous()
        x = x.view(x.shape[0], x.shape[1], -1)
        x = F.dropout(self.fc1(x), p=0.5)
        output, _ = self.gru1(x)
        x = self.fc2(output)
        x = x.permute(1, 0, 2)

        return x