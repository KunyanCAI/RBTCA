import torch
import torch.nn as nn
from .modules.layers import DownBlock, Upblock, ConvBatchNorm, AttnUpBlock, ResConvBatchNorm

class UNetSegmentor(nn.Module):
    """
    Standard U-Net segmentation backbone.
    Adapted to accept multi-channel input (RGB + ROI).
    """
    def __init__(self, in_channels, n_classes, base_channel=64):
        super().__init__()
        # Initial convolution layer
        self.inc = ConvBatchNorm(in_channels, base_channel)
        
        # Encoder path
        self.down1 = DownBlock(base_channel, base_channel * 2, nb_Conv=2)
        self.down2 = DownBlock(base_channel * 2, base_channel * 4, nb_Conv=2)
        self.down3 = DownBlock(base_channel * 4, base_channel * 8, nb_Conv=2)
        self.down4 = DownBlock(base_channel * 8, base_channel * 8, nb_Conv=2)
        
        # Decoder path
        self.up4 = Upblock(base_channel * 16, base_channel * 4, nb_Conv=2)
        self.up3 = Upblock(base_channel * 8, base_channel * 2, nb_Conv=2)
        self.up2 = Upblock(base_channel * 4, base_channel, nb_Conv=2)
        self.up1 = Upblock(base_channel * 2, base_channel, nb_Conv=2)
        
        # Output layer
        self.outc = nn.Conv2d(base_channel, n_classes, kernel_size=1)
        self.last_activation = nn.Sigmoid()
        self.n_classes = n_classes

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        x = self.up4(x5, x4)
        x = self.up3(x, x3)
        x = self.up2(x, x2)
        x = self.up1(x, x1)
        
        logits = self.outc(x)
        if self.n_classes == 1:
            logits = self.last_activation(logits)
        return logits

class AttnUNetSegmentor(nn.Module):
    """
    Attention U-Net segmentation backbone.
    Uses Attention Gates in the decoder path.
    """
    def __init__(self, in_channels, n_classes, base_channel=64):
        super().__init__()
        # Initial convolution layer
        self.inc = ConvBatchNorm(in_channels, base_channel)
        
        # Encoder path
        self.down1 = DownBlock(base_channel, base_channel * 2, nb_Conv=2)
        self.down2 = DownBlock(base_channel * 2, base_channel * 4, nb_Conv=2)
        self.down3 = DownBlock(base_channel * 4, base_channel * 8, nb_Conv=2)
        self.down4 = DownBlock(base_channel * 8, base_channel * 8, nb_Conv=2)
        
        # Decoder path with Attention Gates
        self.up4 = AttnUpBlock(base_channel * 16, base_channel * 4, nb_Conv=2)
        self.up3 = AttnUpBlock(base_channel * 8, base_channel * 2, nb_Conv=2)
        self.up2 = AttnUpBlock(base_channel * 4, base_channel, nb_Conv=2)
        self.up1 = AttnUpBlock(base_channel * 2, base_channel, nb_Conv=2)
        
        # Output layer
        self.outc = nn.Conv2d(base_channel, n_classes, kernel_size=1)
        self.last_activation = nn.Sigmoid()
        self.n_classes = n_classes

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        x = self.up4(x5, x4)
        x = self.up3(x, x3)
        x = self.up2(x, x2)
        x = self.up1(x, x1)
        
        logits = self.outc(x)
        if self.n_classes == 1:
            logits = self.last_activation(logits)
        return logits

class ResUNetSegmentor(nn.Module):
    """
    Residual U-Net segmentation backbone.
    Uses Residual Blocks instead of standard Convolution Blocks.
    """
    def __init__(self, in_channels, n_classes, base_channel=64):
        super().__init__()
        # Initial convolution layer
        self.inc = ResConvBatchNorm(in_channels, base_channel)
        
        # Encoder path
        self.down1 = DownBlock(base_channel, base_channel * 2, nb_Conv=2, block=ResConvBatchNorm)
        self.down2 = DownBlock(base_channel * 2, base_channel * 4, nb_Conv=2, block=ResConvBatchNorm)
        self.down3 = DownBlock(base_channel * 4, base_channel * 8, nb_Conv=2, block=ResConvBatchNorm)
        self.down4 = DownBlock(base_channel * 8, base_channel * 8, nb_Conv=2, block=ResConvBatchNorm)
        
        # Decoder path
        self.up4 = Upblock(base_channel * 16, base_channel * 4, nb_Conv=2, block=ResConvBatchNorm)
        self.up3 = Upblock(base_channel * 8, base_channel * 2, nb_Conv=2, block=ResConvBatchNorm)
        self.up2 = Upblock(base_channel * 4, base_channel, nb_Conv=2, block=ResConvBatchNorm)
        self.up1 = Upblock(base_channel * 2, base_channel, nb_Conv=2, block=ResConvBatchNorm)
        
        # Output layer
        self.outc = nn.Conv2d(base_channel, n_classes, kernel_size=1)
        self.last_activation = nn.Sigmoid()
        self.n_classes = n_classes

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        x = self.up4(x5, x4)
        x = self.up3(x, x3)
        x = self.up2(x, x2)
        x = self.up1(x, x1)
        
        logits = self.outc(x)
        if self.n_classes == 1:
            logits = self.last_activation(logits)
        return logits