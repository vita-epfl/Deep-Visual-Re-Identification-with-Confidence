from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .cross_entropy_loss import CrossEntropyLabelSmooth, AngularLabelSmooth,AdaptiveLabelSmooth,LabelSmooth_sigmoid,AdaptiveLabelSmooth_sigmoid, modifiedBCE
from .hard_mine_triplet_loss import TripletLoss,SoftTripletLoss
from .angular_softmax import AngleLoss

from .entropy_loss import ConfidencePenalty
from .customTripletLoss import TripletLoss_custom,SoftTripletLoss_custom
from .loss_autotune import MultiHeadLossAutoTune
from .info_loss import InfoLoss
from .jsd import JSD_loss
from .focal_loss import FocalLoss


def DeepSupervision(criterion, xs, y):
    """
    Args:
    - criterion: loss function
    - xs: tuple of inputs
    - y: ground truth
    """
    loss = 0.
    for x in xs:
        loss += criterion(x, y)
    loss /= len(xs)
    return loss


def DeepSupervisionAdaptive(criterion, xs, y,epsilon):
    """
    Args:
    - criterion: loss function
    - xs: tuple of inputs
    - y: ground truth
    """
    loss = 0.
    for x in xs:
        loss += criterion(x, y,epsilon)
    loss /= len(xs)
    return loss
