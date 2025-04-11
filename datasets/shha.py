# -*- coding: utf-8 -*-

import numpy as np
import os
import torch
import torch.nn.functional as tF
from torch.utils import data
from PIL import Image
import tqdm

if __name__ == '__main__':
    from utils import NormalSample
else:
    from .utils import NormalSample


class SHHA(data.Dataset):
    def __init__(self, root_path, mode, move_img_to_memory):
        self.imgids = []
        # ------- qnrf_patches --------
        # imtype = 'jpg'
        # for imgf in os.listdir(os.path.join(root_path, mode)):
        #     if imtype in imgf:
        #         self.imgids.append(imgf.replace(f'.{imtype}', ''))
        
        # self.imgpath = os.path.join(root_path, mode, '{}' + f'.{imtype}')
        # self.dotpath = os.path.join(root_path, mode, '{}_ann_new.npy')

        # ------- QNRF_SE_1536 -------
        imtype = 'png'
        for imgf in os.listdir(os.path.join(root_path, mode)):
            if imtype in imgf:
                self.imgids.append(imgf.replace(f'.{imtype}', ''))
        
        self.imgpath = os.path.join(root_path, mode, '{}' + f'.{imtype}')
        self.dotpath = os.path.join(root_path, mode, '{}.npy')
        
        self.mode = mode
        self.normalfunc = NormalSample(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            train = (mode == 'train')
        )
    
    def __len__(self):
        return len(self.imgids)

    def __getitem__(self, index):
        smpid = self.imgids[index]
        img, dotseq = self.readSampleFromId(smpid, resize_factor=1)
        image, dotseq = self.normalfunc(img, dotseq)
        # print(image.shape, len(dotseq))
        return image, dotseq, smpid


    def readSampleFromId(self, smpid, resize_factor=1):
        imgpath = self.imgpath.format(smpid)
        img = Image.open(imgpath).convert('RGB')
        if resize_factor > 1:
            img = img.resize((img.width*resize_factor, img.height*resize_factor), Image.LANCZOS)
        
        img = self.normalfunc.im2tensor(img)
        dotseq = torch.from_numpy(np.load(self.dotpath.format(smpid)))[:, :2] * resize_factor
        
        return img, dotseq

    @staticmethod
    def collate_fn(samples):
        images, seqinfo, imgnames = zip(*samples)
        images = torch.cat(images, dim=0)
        seqinfo = sum(seqinfo, [])
        return images, seqinfo, imgnames

class DeNormalize(object):
    def __init__(self, mean, std):
        self.mean = torch.Tensor(mean)
        self.std = torch.Tensor(std)

    def __call__(self, tensor):
        mean = self.mean.to(tensor.device).view(1, 3, 1, 1)
        std = self.std.to(tensor.device).view(1, 3, 1, 1)
        return tensor * std + mean

if __name__ == '__main__':
    datadir = "/qnap/home_archive/wlin38/crowd/data/ori_data/ShanghaiTech/part_B"
    data = SHHA(datadir, 'train', False)
    denormal = DeNormalize(
         mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    import cv2
    for j, (imgs, dseqs, sid) in enumerate(data):
        print("B:", imgs.shape, len(dseqs))
        # break
        for i in range(imgs.shape[0]):
            image, dseq = imgs[0], dseqs[i]
            image = denormal(image).squeeze() * 255.
            print(image.shape)
            img = image.numpy().transpose((1, 2, 0))[:, :, ::-1].astype(np.uint8).copy()
            print(f"img[{i}]", img.shape, img.dtype, dseq.shape)
            
            # cv2.imwrite(f'image[{i}].png', img)
            print(dseq.amax(dim=0), img.shape)
            for i, dot in enumerate(dseq.int()):
                y, x, r = [a.item() for a in dot]
                # print(i, dot)
                img = cv2.circle(img, (x, y), r, (0, 0, 255), 1)
            cv2.imwrite(f'image{sid}.png', img)
            break
        if j >= 1:
            break
