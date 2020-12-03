from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .focal_loss import FocalLoss
from .label_smoothing import LabelSmoothingCrossEntropy
from .dice_loss import DiceLoss
from .autoweighted_loss import AutomaticWeightedLoss

__all__ = [
    'FocalLoss',
    'LabelSmoothingCrossEntropy',
    'DiceLoss',
    'AutomaticWeightedLoss'
]