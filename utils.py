# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------

import os
import torch
import torch.distributed as dist
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import random

def load_checkpoint(config, model, optimizer, lr_scheduler, logger):
    logger.info(f"==============> Resuming form {config.MODEL.RESUME}....................")
    if config.MODEL.RESUME.startswith('https'):
        checkpoint = torch.hub.load_state_dict_from_url(
            config.MODEL.RESUME, map_location='cpu', check_hash=True)
    else:
        checkpoint = torch.load(config.MODEL.RESUME, map_location='cpu')
    msg = model.load_state_dict(checkpoint['model'], strict=False)
    logger.info(msg)
    max_accuracy = [1e6] * 3
    return max_accuracy
    if not config.EVAL_MODE and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
        #lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        config.defrost()
        config.TRAIN.START_EPOCH = checkpoint['epoch'] + 1
        config.freeze()
        logger.info(f"=> loaded successfully '{config.MODEL.RESUME}' (epoch {checkpoint['epoch']})")
        if 'max_accuracy' in checkpoint:
            max_accuracy = checkpoint['max_accuracy']

    del checkpoint
    torch.cuda.empty_cache()
    return max_accuracy


def save_checkpoint(config, epoch, model, max_accuracy, logger):
    save_state = {
        'model': model.state_dict(),
        'max_accuracy': max_accuracy,
        'config': config
    }
    save_path = os.path.join(config.OUTPUT, f'ckpt_epoch_{epoch}.pth')
    logger.info(f"{save_path} saving......")
    torch.save(save_state, save_path)
    logger.info(f"{save_path} saved !!!")


def get_grad_norm(parameters, norm_type=2):
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    norm_type = float(norm_type)
    total_norm = 0
    for p in parameters:
        param_norm = p.grad.data.norm(norm_type)
        total_norm += param_norm.item() ** norm_type
    total_norm = total_norm ** (1. / norm_type)
    return total_norm


def auto_resume_helper(output_dir):
    checkpoints = os.listdir(output_dir)
    checkpoints = [ckpt for ckpt in checkpoints if ckpt.endswith('pth')]
    print(f"All checkpoints founded in {output_dir}: {checkpoints}")
    if len(checkpoints) > 0:
        latest_checkpoint = max([os.path.join(output_dir, d) for d in checkpoints], key=os.path.getmtime)
        print(f"The latest checkpoint founded: {latest_checkpoint}")
        resume_file = latest_checkpoint
    else:
        resume_file = None
    return resume_file

import os
import matplotlib.pyplot as plt

class CurvePlotter:
    def __init__(self, label, savedir):
        self.savedir = savedir
        self.label = label
        self.epo = []  # Initialize epochs list
        self.data = []  # Initialize data list
        self.key_epoch = None  # Initialize key epoch
        self.savepath = os.path.join(savedir, f'{label}_curve.png')

    def plot_curve(self):
        """
        Plots the curve with epochs on x-axis and data on y-axis.
        If a key epoch is set, marks it with a red dot.
        Saves the figure to the specified path and closes it.
        """
        if not self.epo or not self.data:
            raise ValueError("Both 'epo' and 'data' lists must have values to plot.")
        
        fig = plt.figure()
        plt.title(self.label)
        plt.plot(self.epo, self.data, label='Data Curve')
        
        # Mark key epoch if set
        if self.key_epoch is not None and self.key_epoch in self.epo:
            key_index = self.epo.index(self.key_epoch)
            plt.plot(self.epo[key_index], self.data[key_index], 'ro', label='best epoch')
            plt.legend()

        plt.xlabel('Epochs')
        plt.ylabel(self.label)
        plt.grid(True)
        plt.savefig(self.savepath)
        plt.close(fig)

    def add(self, epoch, value):
        """
        Adds a new data point to the 'epo' and 'data' lists.
        """
        self.epo.append(epoch)
        self.data.append(value)

    def set_key_epoch(self, epoch):
        """
        Sets the key epoch.
        """
        self.key_epoch = epoch

    def add_and_plot(self, epoch, value):
        self.add(epoch, value)
        self.plot_curve()

def set_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True