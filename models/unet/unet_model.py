""" Full assembly of the parts to form the complete network """

from .unet_parts import *


class UNet(nn.Module):
    def __init__(self, n_channels=1, n_classes=2, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)
        
        self.Softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # x:[1, d, h, w]
        x1 = self.inc(x)        # [1, 64, d, h, w]
        x2 = self.down1(x1)     # [1, 128, d/2, h/2, w/2]
        x3 = self.down2(x2)     # [1, 256, d/4, h/4, w/4]
        x4 = self.down3(x3)     # [1, 512, d/8, h/8, w/8]
        x5 = self.down4(x4)     # [1, 1024, d/16, h/16, w/16]
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        logits = self.Softmax(x)
        return logits
