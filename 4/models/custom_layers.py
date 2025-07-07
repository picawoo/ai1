import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt


class CustomConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.zeros(out_channels))
        nn.init.kaiming_normal_(self.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        weight_norm = self.weight / (self.weight.norm(p=2, dim=(1, 2, 3), keepdim=True) * sqrt(self.weight.size(1)))
        return F.conv2d(x, weight_norm, self.bias, self.stride, self.padding)


class AttentionBlock(nn.Module):
    """Attention механизм для CNN"""

    def __init__(self, channels):
        super().__init__()
        self.query = nn.Conv2d(channels, channels // 8, 1)
        self.key = nn.Conv2d(channels, channels // 8, 1)
        self.value = nn.Conv2d(channels, channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch_size, C, H, W = x.size()
        q = self.query(x).view(batch_size, -1, H * W).permute(0, 2, 1)  # (B, HW, C')
        k = self.key(x).view(batch_size, -1, H * W)  # (B, C', HW)
        v = self.value(x).view(batch_size, -1, H * W)  # (B, C, HW)
        attn = torch.bmm(q, k)  # (B, HW, HW)
        attn = F.softmax(attn, dim=-1)
        out = torch.bmm(v, attn.permute(0, 2, 1))
        out = out.view(batch_size, C, H, W)

        return self.gamma * out + x


class CustomActivation(nn.Module):
    """Кастомная активация Mish: x * tanh(softplus(x))"""

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * torch.tanh(F.softplus(x))


class CustomPooling(nn.Module):
    """Кастомный пулинг с learnable весами"""

    def __init__(self, pool_size=2):
        super().__init__()
        self.pool_size = pool_size
        self.weights = nn.Parameter(torch.ones(1, 1, pool_size, pool_size))

    def forward(self, x):
        weights = F.softmax(self.weights.view(-1), dim=0).view_as(self.weights)
        return F.avg_pool2d(x * weights, self.pool_size, divisor_override=1)