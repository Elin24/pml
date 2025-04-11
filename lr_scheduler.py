# -*- coding: utf-8 -*-

import torch
from torch.optim.lr_scheduler import LambdaLR, LRScheduler, StepLR, CosineAnnealingLR, ConstantLR



class WarmUpLRScheduler():
    def __init__(
        self,
        warm_up_epochs: int,
        warm_up_lr: float,
        normal_lr_scheduler: LRScheduler
    ):
        optimizer = normal_lr_scheduler.optimizer
        self.warmup_epochs = warm_up_epochs
        if warm_up_epochs > 0:
            warmup_end_lr = optimizer.param_groups[0]['lr']
            if isinstance(warmup_end_lr, torch.Tensor):
                warmup_end_lr = warmup_end_lr.item()
            
            warmup_lr_scale = warm_up_lr / warmup_end_lr
            # self.warmup_lrs = [ base_lr * warmup_lr_scale for base_lr in normal_lr_scheduler.base_lrs ]
            # self.warmup_gamma = warmup_lr_scale ** (- 1 / (warm_up_epochs))
            
            # for param_group, warmup_lr in zip(optimizer.param_groups, self.warmup_lrs):
            #     param_group["lr"] = warmup_lr
            # self.warmup_scheduler = StepLR(optimizer, step_size=1, gamma=self.warmup_gamma)
            
            self.warmup_scheduler = LambdaLR(optimizer, lr_lambda=lambda step: self.warmup_linear(step, warmup_lr_scale))

        else:
            self.warmup_scheduler = None
        self.normal_scheduler = normal_lr_scheduler
        
        # print(self.get_lr())
        self.last_epoch = 0
        # super().__init__(optimizer, last_epoch, verbose)

    def step(self, epoch = None):
        self.last_epoch = (self.last_epoch + 1) if epoch is None else epoch

        if self.warmup_scheduler is None or self.last_epoch > self.warmup_epochs:
            self.normal_scheduler.step()
        else:
            self.warmup_scheduler.step()
        # print(self.last_epoch, self.warmup_scheduler.last_epoch, self.normal_scheduler.last_epoch)

    def warmup_linear(self, current_step: int, warm_up_scale):
        if current_step < self.warmup_epochs:
            scale_factor = current_step / self.warmup_epochs * (1 - warm_up_scale) + warm_up_scale
            return scale_factor
        else:
            return 1.0


def build_scheduler(optimizer, train_config, n_iter_per_epoch):
    warm_up_iterns = train_config.WARMUP_EPOCHS * n_iter_per_epoch
    train_iterns = (train_config.EPOCHS - train_config.WARMUP_EPOCHS) * n_iter_per_epoch
    
    if train_config.LR_SCHEDULER.NAME == "step":
        DECAY_RATE = (train_config.BASE_LR / train_config.MIN_LR) ** (- 1 / train_iterns)
        scheduler = StepLR(
            optimizer = optimizer, 
            step_size = 1,
            gamma = DECAY_RATE
        )
    elif train_config.LR_SCHEDULER.NAME == 'cosine':
        scheduler = CosineAnnealingLR(
            optimizer = optimizer,
            T_max = train_iterns,
            eta_min = train_config.MIN_LR
        )
    elif train_config.LR_SCHEDULER.NAME == 'constant':
        scheduler = ConstantLR(
            optimizer, factor=1, total_iters=0
        )
    else:
        raise ValueError("Unknown LR scheduler name: {}".format(train_config.LR_SCHEDULER.NAME))


    return WarmUpLRScheduler(
        warm_up_epochs = warm_up_iterns,
        warm_up_lr = train_config.WARMUP_LR,
        normal_lr_scheduler = scheduler
    )