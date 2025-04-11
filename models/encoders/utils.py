import torch
import torch.nn.init as init
import torch.nn as nn
import torch.nn.functional as tF

def convblock(inc, ouc, kernel_size, bn=True):
    padding = kernel_size // 2
    module = nn.Sequential(
        nn.Conv2d(inc, ouc, kernel_size=kernel_size, stride=1, padding=padding, bias=not bn),
        nn.BatchNorm2d(ouc) if bn else nn.Identity(),
        nn.ReLU(inplace=True)
    )
    if not bn: 
        init.constant_(module[0].bias, 0.)

    return module

def conv_3x3(inc, ouc, bn=True):
    return convblock(inc, ouc, kernel_size=3, bn=bn)

class UpSample_P2P(nn.Module):
    def __init__(self, incs, ouc, bn=True):
        super().__init__()
        inc1, inc2 = incs

        self.conv_small = nn.Conv2d(inc1, ouc, kernel_size=1, stride=1, padding=0, bias=not bn)
        self.conv_large = nn.Conv2d(inc2, ouc, kernel_size=1, stride=1, padding=0, bias=not bn)
        if not bn:
            init.constant_(self.conv_large.bias, 0.)
            init.constant_(self.conv_small.bias, 0.)

        self.fuse = nn.Sequential(
            nn.Conv2d(ouc, ouc, kernel_size=3, stride=1, padding=1, bias= not bn),
            nn.BatchNorm2d(ouc) if bn else nn.Identity(),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        xs, xl = x
        xs = self.conv_small(xs)
        xs = tF.interpolate(xs, xl.shape[-2:], mode='bilinear', align_corners=False)
        xl = self.conv_large(xl)
        x = self.fuse(xs + xl)
        return x

class UpSample_P2PS(nn.Module):
    def __init__(self, incs, out_channel_factor=2, bn=True, post_process=True):
        super().__init__()

        incs = incs[::-1]

        self.fuse_layers = nn.ModuleList()
        last_out = incs[0]
        for inc in incs[1:]:
            self.fuse_layers.append(UpSample_P2P([last_out, inc], inc // out_channel_factor, bn=bn))
            last_out = inc // out_channel_factor
        
        self.fuse_channel = last_out

        self.post_process = conv_3x3(last_out, last_out, bn=bn) if post_process else nn.Identity()
    
    def forward(self, xs):
        xs = xs[::-1]
        x0, xs = xs[0], xs[1:]
        for x, fuse_layer in zip(xs, self.fuse_layers):
            x0 = fuse_layer([x0, x])
        return self.post_process(x0)
