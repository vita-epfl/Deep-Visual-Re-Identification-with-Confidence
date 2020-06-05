from __future__ import absolute_import
from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F

def one_hot_embedding(labels, num_classes):
    return torch.eye(num_classes)[labels.data.cpu()].cuda()

class FocalLoss(nn.Module):

    def __init__(self, gamma=0, alpha=1, eps=1e-7):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.eps = eps
        self.alpha = alpha

    def forward(self, inputs, targets):
        #targets = torch.zeros(inputs.size()).scatter_(1, targets.unsqueeze(1).data.cpu(), 1)
        ce_loss = F.cross_entropy(inputs, targets, reduction='none') # important to add reduction='none' to keep per-batch-item loss
        pt = torch.exp(-ce_loss)
        import pdb; pdb.set_trace()
        focal_loss = (self.alpha * (1-pt)**self.gamma * ce_loss).mean()

        return focal_loss
