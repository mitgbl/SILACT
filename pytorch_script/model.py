from util import down, up, outconv
import torch.nn as nn

''' Unet Design'''
class UNet_3(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet_3, self).__init__()
#        self.inc = inconv(n_channels, 4,(3,3))
        self.down1 = down(n_channels, 8, (5,5))
        self.down2 = down(8, 16, (5,5))
        self.down3 = down(16, 32, (5,5))
        self.down4 = down(32, 64, (5,5))
        self.down5 = down(64, 128, (5,5))
        self.up1 = up(128,64,200,(5,5))
        self.up2 = up(200,32,180,(5,5))
        self.up3 = up(180,16,160,(5,5))
        self.up4 = up(160,8,140,(5,5))
        self.up5 = up(140,n_channels,120,(5,5))
        self.outc = outconv(120, n_classes,(5,5))

    def forward(self, x):
#        x1 = self.inc(x)
        x0 = x
        x1 = self.down1(x0)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        x5 = self.down5(x4)
        x = self.up1(x5,x4)
        x = self.up2(x,x3)
        x = self.up3(x,x2)
        x = self.up4(x,x1)
        x = self.up5(x,x0)
        x = self.outc(x)
        return x