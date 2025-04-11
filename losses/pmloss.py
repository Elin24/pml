# -*- coding: utf-8 -*-

import torch
import torch.nn as nn

eps = 1e-10
class L2DIS:
    def __init__(self, factor=512) -> None:
        self.factor = factor

    def __call__(self, X, Y):
        x_col = X.unsqueeze(-2)
        y_row = Y.unsqueeze(-3)
        C = torch.norm(x_col - y_row, dim=-1)
        C = C / self.factor
        return C

class PMLoss(nn.modules.loss._Loss):
    def __init__(self, factor=1) -> None:
        super().__init__()
        self.factor = factor
        self.cost = L2DIS(1)
        self.w_co = 1
        self.radis = 32

    def forward(self, dens, seqs, down):
        bs = len(seqs)
        oot_loss, cnt_loss = 0, 0
        for i in range(bs):
            den, seq = dens[i, 0], seqs[i]
            H, W = den.shape
            count = seq[:, -1].sum()
            if count < 1:
               oot_loss = oot_loss + torch.abs(den).sum()
               cnt_loss = cnt_loss + torch.abs(den.sum())
            elif den.sum() > self.factor:
                A_coord = torch.stack(torch.meshgrid(torch.arange(H), torch.arange(W)), dim=-1).view(1, -1, 2)
                A_coord = A_coord.to(seq).float() * down + (down - 1) / 2
                A = den.view(1, -1, 1)
                
                B_coord = seq[None, :, :2].float()
                B = torch.ones((1, B_coord.size(1), 1)).to(A)
                
                with torch.no_grad():
                    C = self.cost(A_coord, B_coord)
                    minC, mcidx = C.min(dim=-1, keepdim=True)
                    M = torch.zeros_like(C).scatter_(-1, mcidx, 1.0)
                    maxC = (minC.view_as(A) * M).amax(dim=1, keepdim=True)
                    maxC = (M * maxC).sum(dim=-1).view_as(A)
                    C = minC / torch.clamp_min(maxC, min=self.radis)
                    C = torch.exp(C) - 1

                    avgC = (C * M).sum(dim=1, keepdim=True) / (M.sum(dim=1, keepdim=True) + eps)
                    avgC = (avgC * M).sum(dim=-1).view_as(C)
                    F = C - avgC

                P = (M * A).sum(dim=1).view_as(B)
                
                oot_loss = oot_loss + torch.sum(A * F.detach()) * self.w_co
                cnt_loss = cnt_loss + torch.abs(P - B).sum() * self.w_co

            else:
                cnt_loss = cnt_loss + torch.abs(den.sum() - count * self.factor) * self.w_co
        loss = (oot_loss + cnt_loss) / bs
        return loss
    

    def den2coord(self, denmap):
        assert denmap.dim() == 2, f"denmap.shape = {denmap.shape}, whose dim is not 2"
        coord = torch.nonzero(denmap)
        denval = denmap[coord[:, 0], coord[:, 1]]
        return denval.view(1, -1, 1), coord.view(1, -1, 2)
