# -*- coding: utf-8 -*-
"""
PMLoss implementation for density map-based crowd counting.

This module defines the Point Matching Loss (PMLoss) which is used to train
crowd counting models by matching predicted density maps to ground truth point sequences.
"""

import torch
import torch.nn as nn

# Small epsilon value to avoid division by zero
eps = 1e-10

class L2DIS:
    """
    L2 distance calculator for coordinate matching.
    
    Args:
        factor (float): Scaling factor for the distance values
    """
    def __init__(self, factor=512) -> None:
        self.factor = factor

    def __call__(self, X, Y):
        """
        Compute L2 distances between all pairs of points in X and Y.
        
        Args:
            X: Tensor of shape (..., N, 2) containing coordinates
            Y: Tensor of shape (..., M, 2) containing coordinates
            
        Returns:
            Tensor of shape (..., N, M) containing scaled L2 distances
        """
        # Expand dimensions for broadcasting
        x_col = X.unsqueeze(-2)  # Shape: (..., 1, N, 2)
        y_row = Y.unsqueeze(-3)  # Shape: (..., M, 1, 2)
        
        # Compute L2 distance between all pairs
        C = torch.norm(x_col - y_row, dim=-1)  # Shape: (..., N, M)
        
        # Scale distances
        C = C / self.factor
        return C

class PMLoss(nn.modules.loss._Loss):
    """
    Point Matching Loss for crowd counting models.
    
    Args:
        factor (float): Scaling factor for density map values
    """
    def __init__(self, factor=1) -> None:
        super().__init__()
        self.factor = factor
        self.cost = L2DIS(1)  # L2 distance calculator with no scaling
        self.w_co = 1  # Weight for loss components
        self.radis = 32  # Radius threshold for distance normalization

    def forward(self, dens, seqs, down):
        """
        Compute the point matching loss.
        
        Args:
            dens: Tensor of predicted density maps
            seqs: List of ground truth point sequences
            down: Downsampling factor between input and output
            
        Returns:
            Average loss across the batch
        """
        bs = len(seqs)
        oot_loss, cnt_loss = 0, 0  # Out-of-target loss and count loss
        
        for i in range(bs):
            den, seq = dens[i, 0], seqs[i]  # Get density map and point sequence
            H, W = den.shape  # Height and width of density map
            count = float(seq.shape[0])  # Number of people in the ground truth
            
            if count < 1:
                # No people in the image, penalize any predicted density
                oot_loss = oot_loss + torch.abs(den).sum()
                cnt_loss = cnt_loss + torch.abs(den.sum())
            
            elif den.sum() > self.factor:
                # Sufficient predicted density, perform point matching
                # Generate coordinates for all pixels in the density map
                A_coord = torch.stack(torch.meshgrid(torch.arange(H), torch.arange(W)), dim=-1).view(1, -1, 2)
                A_coord = A_coord.to(seq).float() * down + (down - 1) / 2  # Scale to original image space
                A = den.view(1, -1, 1)  # Flatten density map
                
                # Extract coordinates from ground truth points
                B_coord = seq[None, :, :2].float()
                B = torch.ones((1, B_coord.size(1), 1)).to(A)  # Binary mask for ground truth points
                
                with torch.no_grad():
                    # Compute cost matrix (distances between all predicted and ground truth points)
                    C = self.cost(A_coord, B_coord)
                    
                    # Find closest ground truth point for each predicted point
                    minC, mcidx = C.min(dim=-1, keepdim=True)
                    M = torch.zeros_like(C).scatter_(-1, mcidx, 1.0)  # Matching matrix
                    
                    # Normalize distances
                    maxC = (minC.view_as(A) * M).amax(dim=1, keepdim=True)
                    maxC = (M * maxC).sum(dim=-1).view_as(A)
                    C = minC / torch.clamp_min(maxC, min=self.radis)
                    C = torch.exp(C) - 1

                    # Compute average cost for each predicted point
                    avgC = (C * M).sum(dim=1, keepdim=True) / (M.sum(dim=1, keepdim=True) + eps)
                    avgC = (avgC * M).sum(dim=-1).view_as(C)
                    F = C - avgC  # Cost difference from average

                # Compute predicted count for each ground truth point
                P = (M * A).sum(dim=1).view_as(B)
                
                # Update loss components
                oot_loss = oot_loss + torch.sum(A * F.detach()) * self.w_co
                cnt_loss = cnt_loss + torch.abs(P - B).sum() * self.w_co

            else:
                # Low predicted density, just penalize count error
                cnt_loss = cnt_loss + torch.abs(den.sum() - count * self.factor) * self.w_co
        
        # Average loss across batch
        loss = (oot_loss + cnt_loss) / bs
        return loss
    

    def den2coord(self, denmap):
        """
        Convert density map to coordinate points.
        
        Args:
            denmap: 2D density map tensor
            
        Returns:
            Tuple of (density values, coordinates) for non-zero points
        """
        assert denmap.dim() == 2, f"denmap.shape = {denmap.shape}, whose dim is not 2"
        coord = torch.nonzero(denmap)  # Get coordinates of non-zero points
        denval = denmap[coord[:, 0], coord[:, 1]]  # Get density values at these coordinates
        return denval.view(1, -1, 1), coord.view(1, -1, 2)
