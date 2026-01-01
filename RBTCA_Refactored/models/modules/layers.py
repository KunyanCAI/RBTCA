import torch.nn as nn
import torch

def get_activation(activation_type):
    activation_type = activation_type.lower()
    if hasattr(nn, activation_type):
        return getattr(nn, activation_type)()
    else:
        return nn.ReLU()

class ConvBatchNorm(nn.Module):
    """(convolution => [BN] => ReLU)"""
    def __init__(self, in_channels, out_channels, activation='ReLU'):
        super(ConvBatchNorm, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm = nn.BatchNorm2d(out_channels)
        self.activation = get_activation(activation)

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        return self.activation(out)

class ResConvBatchNorm(nn.Module):
    """Residual Block: x + (convolution => [BN] => ReLU => convolution => [BN]) => ReLU"""
    def __init__(self, in_channels, out_channels, activation='ReLU'):
        super(ResConvBatchNorm, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = get_activation(activation)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        if in_channels != out_channels:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.residual = nn.Identity()

    def forward(self, x):
        res = self.residual(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += res
        return self.relu(out)

class AttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(AttentionGate, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi

def _make_nConv(in_channels, out_channels, nb_Conv, activation='ReLU', block=ConvBatchNorm):
    layers = []
    layers.append(block(in_channels, out_channels, activation))
    for _ in range(nb_Conv - 1):
        layers.append(block(out_channels, out_channels, activation))
    return nn.Sequential(*layers)

class DownBlock(nn.Module):
    """Downscaling with maxpool convolution"""
    def __init__(self, in_channels, out_channels, nb_Conv, activation='ReLU', block=ConvBatchNorm):
        super(DownBlock, self).__init__()
        self.maxpool = nn.MaxPool2d(2)
        self.nConvs = _make_nConv(in_channels, out_channels, nb_Conv, activation, block)

    def forward(self, x):
        out = self.maxpool(x)
        return self.nConvs(out)

class Upblock(nn.Module):
    def __init__(self, in_channels, out_channels, nb_Conv, activation='ReLU', block=ConvBatchNorm):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2)
        # in_channels here is the combined channels (skip + upsampled)
        self.nConvs = _make_nConv(in_channels, out_channels, nb_Conv, activation, block)

    def forward(self, x, skip_x):
        up = self.up(x)
        x = torch.cat([skip_x, up], dim=1)
        return self.nConvs(x)

class AttnUpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, nb_Conv, activation='ReLU', block=ConvBatchNorm):
        super(AttnUpBlock, self).__init__()
        self.up = nn.Upsample(scale_factor=2)
        
        # in_channels is the size AFTER concatenation.
        # Assuming symmetric U-Net where skip and upsampled features have equal channels = in_channels // 2
        skip_channels = in_channels // 2
        gating_channels = in_channels // 2
        
        self.attn = AttentionGate(F_g=gating_channels, F_l=skip_channels, F_int=skip_channels // 2)
        self.nConvs = _make_nConv(in_channels, out_channels, nb_Conv, activation, block)

    def forward(self, x, skip_x):
        up = self.up(x)
        skip_x = self.attn(g=up, x=skip_x)
        x = torch.cat([skip_x, up], dim=1)
        return self.nConvs(x)
