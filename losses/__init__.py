# -*- coding: utf-8 -*-

from .pmloss import PMLoss

def build_loss(config):
    factor = config.FACTOR
    lossfunc = config.LOSS

    Loss = {
        'PML': PMLoss,
    }[lossfunc]

    return Loss(factor=factor), Loss(factor=factor)
