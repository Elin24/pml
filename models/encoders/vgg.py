# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

if __name__ == '__main__':
    from utils import UpSample_P2PS
else:
    from .utils import UpSample_P2PS


class VGGFPN(nn.Module):
    def __init__(self, vgg):
        super().__init__()
        self.last_scale = 8
        self.fuse_layer_num = 2
        self.encoder, self.num_channels = self.load_encoder(vgg, self.last_scale)
        # print(self.num_channels)
        self.fuse_layer = UpSample_P2PS(self.num_channels[:self.fuse_layer_num], out_channel_factor=1, bn=False, post_process=False)
        self.last_channel = self.fuse_layer.fuse_channel
        

    def forward(self, x):
        feas = []
        for module in self.encoder:
            feas.append(x := module(x))
        fea = self.fuse_layer(feas[:self.fuse_layer_num])
        return fea

    def load_encoder(self, vgg_name, last_stride):
        vgg_name = vgg_name.lower()
        assert 'vgg' in vgg_name, f"encoders/vgg.py expects VGG but {vgg_name} is required"
        model_loader = getattr(models, vgg_name)
        weights = getattr(models, f"{vgg_name.upper()}_Weights").DEFAULT
        vgg = model_loader(weights=weights)

        layers = list(vgg.features.children())
        
        encoder, slash = nn.ModuleList(), 0
        num_channels, current_channel = [], 3

        current_stride = 1
        for lid, layer in enumerate(layers):
            if isinstance(layer, nn.MaxPool2d) and current_stride >= last_stride:
                encoder.append(nn.Sequential(*layers[slash:lid]))
                num_channels.append(current_channel)

                slash = lid
            
            if isinstance(layer, nn.Conv2d):
                current_channel = layer.out_channels
            
            if hasattr(layer, 'stride'):
                 current_stride = current_stride * (layer.stride[0] if isinstance(layer.stride, tuple) else layer.stride)

        # encoder.append(nn.Sequential(*layers[slash:]))
        # num_channels.append(current_channel)

        return encoder, num_channels


if __name__ == '__main__':
    model = VGGFPN('vgg19').cuda()
    a = torch.randn(1, 3, 512, 512).cuda()
    xs = model(a)
    for i, x in enumerate(xs):
        print(f"output[{i}]:", x.shape)
    