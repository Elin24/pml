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
        self.fuse_layer_num = 2
        self.encoder, self.num_channels, self.block_strides = self.load_encoder(vgg)
        # fuse only last {fuse_layer_num} channels
        self.fuse_layer = UpSample_P2PS(self.num_channels[-self.fuse_layer_num:], out_channel_factor=1, bn=False, post_process=False)
        self.last_channel = self.fuse_layer.fuse_channel
        self.last_scale = self.block_strides[-self.fuse_layer_num]

    def forward(self, x):
        feas = []
        for module in self.encoder:
            feas.append(x := module(x))
        fea = self.fuse_layer(feas[-self.fuse_layer_num:])
        return fea

    def load_encoder(self, vgg_name):
        vgg_name = vgg_name.lower()
        
        # Get model loader and pretrained weights
        model_loader = getattr(models, vgg_name)
        weights = getattr(models, f"{vgg_name.upper()}_Weights").DEFAULT
        vgg = model_loader(weights=weights)

        # Extract feature layers from the VGG model
        layers = list(vgg.features.children())
        
        # Initialize encoder components
        encoder, slash = nn.ModuleList(), 0  # encoder: list of layer groups, slash: start index of current group
        num_channels, current_channel = [], 3  # num_channels: output channels per group, current_channel: current input channel
        block_strides = []
        current_stride = 1
        for lid, layer in enumerate(layers):
            # When reaching max pooling layer and desired stride, create a new encoder block
            if isinstance(layer, nn.MaxPool2d):
                encoder.append(nn.Sequential(*layers[slash:lid]))
                num_channels.append(current_channel)
                block_strides.append(current_stride)
                slash = lid
            
            # Update current channel count when encountering a convolutional layer
            if isinstance(layer, nn.Conv2d):
                current_channel = layer.out_channels
            
            # Update current stride when layer has stride attribute
            if hasattr(layer, 'stride'):
                 current_stride = current_stride * (layer.stride[0] if isinstance(layer.stride, tuple) else layer.stride)

        return encoder, num_channels, block_strides


if __name__ == '__main__':
    model = VGGFPN('vgg16')# .cuda()
    a = torch.randn(1, 3, 128, 128) # .cuda()
    x = model(a)
    print("output:", x.shape)
    print("last_scale:", model.last_scale)
    