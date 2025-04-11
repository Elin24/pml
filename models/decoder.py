# -*- coding: utf-8 -*-

import torch
import torch.nn.init as init
import torch.nn as nn

if __name__ == '__main__':
    from encoders import conv_3x3
else:
    from .encoders import conv_3x3

class SimpleDecoder(nn.Sequential):
    def __init__(self, in_channel = 128, fea_channel=64, out_channel=1, up_sample=1):
        super().__init__(
            conv_3x3(in_channel, fea_channel, bn=False),
            conv_3x3(fea_channel, fea_channel, bn=False),
            nn.Conv2d(fea_channel, out_channel * (up_sample ** 2), kernel_size=1, stride=1),
            nn.PixelShuffle(up_sample) if up_sample > 1 else nn.Identity()
        )
        init.constant_(self[-2].bias, 0.)
        self[-2].weight.data /= 100

if __name__ == '__main__':
    decoder = SimpleDecoder(512).cuda()
    fea = torch.randn(1, 512 ,32, 32).cuda()
    den = decoder(fea)
    print(den.shape)