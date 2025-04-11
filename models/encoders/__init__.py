# -*- coding: utf-8 -*-

from .vgg import VGGFPN
from .hrnet import HRNetFPN

from .utils import conv_3x3

def build_encoder(name):
    name = name.lower()
    if name.startswith("vgg"):
        return VGGFPN(name)
    elif name.startswith("hrnet"):
        return HRNetFPN(name)
    else:
        raise f"no encoder named {name}"