import sys
from torch.utils import data
print(sys.executable)

import os
import time
import argparse
import datetime
import numpy as np
import torch
from torch import optim
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from timm.utils import AverageMeter

from config import get_config
from models import build_model
from datasets import build_loader
from losses import build_loss
from lr_scheduler import build_scheduler
from logger import create_logger
from utils import load_checkpoint, save_checkpoint, get_grad_norm, auto_resume_helper, CurvePlotter, set_seed

def get_args_parser():
    parser = argparse.ArgumentParser('Counting Everything training and evaluation script', add_help=False)
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )

    # easy config modification
    parser.add_argument('--batch-size', type=int, help="batch size for single GPU")
    parser.add_argument('--data-path', type=str, help='path to dataset')
    parser.add_argument('--resume', help='resume from checkpoint')
    parser.add_argument('--use-checkpoint', action='store_true',
                        help="whether to use gradient checkpointing to save memory")
    parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
    parser.add_argument('--output', default='output', type=str, metavar='PATH',
                        help='root of output folder, the full path is <output>/<model_name>/<tag> (default: output)')
    parser.add_argument('--tag', help='tag of experiment')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--throughput', action='store_true', help='Test throughput only')
    parser.add_argument('--device', default='cuda:0', help='device name')

    args, unparsed = parser.parse_known_args()

    config = get_config(args)

    return args, config

def main_worker(config):
    data_loader_train = build_loader(config.DATA, mode='train')
    data_loader_test = build_loader(config.DATA, mode='test')

    logger.info(f"Creating model with:{config.MODEL.NAME}")
    model = build_model(config.MODEL)
    model.cuda()
    model_without_ddp = model#.module

    criterion, test_criterion = build_loss(config.MODEL)
    criterion.cuda(); test_criterion.cuda()

    param_dicts = [
        {
            "params": [p for n, p in model_without_ddp.named_parameters() if "encoders" not in n and p.requires_grad]
        }, {
            "params": [p for n, p in model_without_ddp.named_parameters() if "encoders" in n and p.requires_grad],
            "lr": config.TRAIN.BACKBONE_LR,
        },
    ]

    optimizer = optim.Adam(param_dicts, lr=config.TRAIN.BASE_LR, weight_decay=config.TRAIN.WEIGHT_DECAY)
    # optimizer = optim.AdamW(param_dicts, lr=config.TRAIN.BASE_LR, weight_decay=config.TRAIN.WEIGHT_DECAY)
    
    lr_scheduler = build_scheduler(optimizer, config.TRAIN, len(data_loader_train))

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"number of params: {n_parameters}")

    

    max_accuracy_test = [1e6] * 2

    if config.TRAIN.AUTO_RESUME:
        resume_file = auto_resume_helper(config.OUTPUT)
        if resume_file:
            if config.MODEL.RESUME:
                logger.warning(f"auto-resume changing resume file from {config.MODEL.RESUME} to {resume_file}")
            config.defrost()
            config.MODEL.RESUME = resume_file
            config.freeze()
            logger.info(f'auto resuming from {resume_file}')
        else:
            logger.info(f'no checkpoint found in {config.OUTPUT}, ignoring auto resume')

    if config.MODEL.RESUME:
        load_checkpoint(config, model_without_ddp, optimizer, lr_scheduler, logger)
        # mae, mse, loss, _ = validate(config, data_loader_val, model, test_criterion)
        # max_accuracy_val = (mae, mse, loss)
        # logger.info(f"Accuracy of the network on the val images: {mae:.2f} | {mse:.2f}")
        # mae, mse, loss, _ = validate(config, data_loader_test, model, test_criterion)
        # max_accuracy_test = (mae, mse, loss)
        # logger.info(f"Accuracy of the network on the test images: {mae:.2f} | {mse:.2f}")
        if config.EVAL_MODE:
            return

    mae_curve = CurvePlotter("mae", os.path.join('exp', config.TAG, 'train.log'))
    mse_curve = CurvePlotter("mse", os.path.join('exp', config.TAG, 'train.log'))

    logger.info("Start training")
    start_time = time.time()
    
    test_feq, last_test = 0, -1
    for epoch in range(config.TRAIN.START_EPOCH, config.TRAIN.EPOCHS + 1):
        
        train_one_epoch(config, model, criterion, data_loader_train, optimizer, lr_scheduler, epoch)
        # save_checkpoint(config, "last_check", model_without_ddp, max_accuracy_val, optimizer, lr_scheduler, logger)
        
        if epoch - last_test  >= test_feq or epoch == (config.TRAIN.EPOCHS):
            mae, mse, _ = validate(config, data_loader_test, model, test_criterion)

            if test_feq > 0:
                mae_curve.add_and_plot(epoch, mae)
                mse_curve.add_and_plot(epoch, mse)

            if mae * 4 + mse < max_accuracy_test[0] * 4 + max_accuracy_test[1]:
                max_accuracy_test = (mae, mse)
                save_checkpoint(config, "best", model_without_ddp, max_accuracy_test, logger)
                if test_feq > 0:
                    mae_curve.set_key_epoch(epoch)
                    mse_curve.set_key_epoch(epoch)

            logger.info(f'Min total MAE|MSE: {max_accuracy_test[0]:.6f} | {max_accuracy_test[1]:.2f}')

            last_test = epoch
            test_feq = max(int(config.MAX_SAVE_FREQ * (config.SAVE_FREQ_FACTOR ** (epoch / config.TRAIN.EPOCHS))), config.MIN_SAVE_FREQ)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info('Training time {}'.format(total_time_str))


def train_one_epoch(config, model, criterion, data_loader, optimizer, lr_scheduler, epoch):
    model.train()
    optimizer.zero_grad()

    num_steps = len(data_loader)
    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    norm_meter = AverageMeter()
    start = time.time()
    end = time.time()
    print_freq = num_steps // max(config.PRINT_FREQ - 1, 1)
    for idx, (samples, dotseq, imgid) in enumerate(data_loader):
        samples = samples.cuda(non_blocking=True)
        dotseq = [d.cuda(non_blocking=True) for d in dotseq]

        denmap = model(samples)
        
        loss = criterion(denmap, dotseq, samples.size(-1) // denmap.size(-1))
        
        optimizer.zero_grad()
        loss.backward()

        grad_norm = get_grad_norm(model.parameters())
        optimizer.step()

        torch.cuda.synchronize()

        if lr_scheduler is not None:
            lr_scheduler.step()

        loss_meter.update(loss.item(), samples.size(0))
        norm_meter.update(grad_norm)
        batch_time.update(time.time() - end)
        end = time.time()

        if idx % print_freq == 0 or idx == num_steps - 1:
            lr = optimizer.param_groups[0]['lr']
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            etas = batch_time.avg * (num_steps - idx)
            logger.info(
                f'Train: [{epoch}/{config.TRAIN.EPOCHS}][{idx}/{num_steps}]\t'
                f'eta {datetime.timedelta(seconds=int(etas))} lr {lr*1e5:.3f}\t'
                f'time {batch_time.val:.4f} ({batch_time.avg:.4f})\t'
                f'loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                f'grad_norm {norm_meter.val  :.3f} ({norm_meter.avg :.3f})\t'
                f'mem {memory_used:.0f}MB')
    epoch_time = time.time() - start
    logger.info(f"EPOCH {epoch} training takes {datetime.timedelta(seconds=int(epoch_time))}")


@torch.inference_mode()
def validate(config, data_loader, model, criterion):
    model.eval()

    batch_time = AverageMeter()
    mae_meter = AverageMeter()
    mse_meter = AverageMeter()

    end = time.time()
    cnts = []
    start_time = end

    num_steps = len(data_loader)
    print_freq = num_steps // max(config.PRINT_FREQ - 1, 1)
    for idx, (images, dotseq, imgid) in enumerate(data_loader):
        images = images.cuda(non_blocking=True)
        dotseq = [d.cuda(non_blocking=True) for d in dotseq]
        cnt = torch.tensor([d.size(0) for d in dotseq]).float().cuda()
        bsize = images.size(0)
        # compute output
        imh, imw = images.shape[-2:]

        output = model(images)
        outnum = output.sum(dim=(1,2,3)) / config.MODEL.FACTOR

        diff = torch.abs(outnum - cnt)
        cnts.append((outnum.item(), cnt.item()))
        mae, mse = diff.mean().item(), (diff ** 2).mean().item()

        mae_meter.update(mae, bsize)
        mse_meter.update(mse, bsize)


        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if idx % print_freq == 0 or idx == num_steps - 1:
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            logger.info(
                f'Test: [{idx}/{len(data_loader)}]  '
                f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})  '
                f'MAE {mae_meter.val:.3f} ({mae_meter.avg:.3f})  '
                f'MSE {mse_meter.val ** 0.5:.3f} ({mse_meter.avg ** 0.5:.3f})  '
                f'Mem {memory_used:.0f}MB')
        
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info(f'Testing {mae_meter.count} samples costs {total_time_str}')

    cnts.sort(key=lambda x: x[1])
    logger.info(f' * MAE {mae_meter.avg:.3f} MSE {mse_meter.avg ** 0.5:.3f}')
    return mae_meter.avg, mse_meter.avg ** 0.5, cnts

if __name__ == '__main__':
    # torch.cuda.set_per_process_memory_fraction(0.5, 0)
    args, config = get_args_parser()

    
    torch.cuda.set_device(args.device)
    set_seed(config.SEED)

    os.makedirs(config.OUTPUT, exist_ok=True)
    logger = create_logger(output_dir=config.OUTPUT, name=f"{config.MODEL.NAME}")

    path = os.path.join(config.OUTPUT, "config.json")
    with open(path, "w") as f:
        f.write(config.dump())
    logger.info(f"Full config saved to {path}")

    # print config
    logger.info(config.dump())

    main_worker(config)
