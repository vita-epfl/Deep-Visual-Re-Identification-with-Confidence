from __future__ import absolute_import
from __future__ import division

import torch
import torch.nn as nn
import math
class InfoLoss(nn.Module):
    """Cross entropy loss with label smoothing regularizer.

    Reference:
    Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.
    Equation: y = (1 - epsilon) * y + epsilon / K.

    Args:
    - num_classes (int): number of classes.
    - epsilon (float): weight.
    """
    def __init__(self, reverse_KL=False):
        super(InfoLoss, self).__init__()
        self.reverse_kl = reverse_KL

    def forward(self, mu, std):
        """
        Args:
        - inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
        - targets: ground truth labels with shape (num_classes)
        """
        if self.reverse_kl:
            loss = -0.5*(1-2*std.log()-(1+mu.pow(2))/(2*std.pow(2))).sum(1).mean().div(math.log(2))
        else:
            loss = -0.5*(1+2*std.log()-mu.pow(2)-std.pow(2)).sum(1).mean().div(math.log(2))
        return loss
