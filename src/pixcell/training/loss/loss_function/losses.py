import torch
import torch.nn as nn

from monai.losses import (
    DiceLoss,
    DiceCELoss,
    FocalLoss,
    GeneralizedDiceLoss,
    TverskyLoss,
)


from pixcell.training.loss.loss_function import TorchLossWrapper
from pixcell.utils.registry import LOSS_REGISTRY


# -------------------------------------------------
# Classification / Segmentation Losses
# -------------------------------------------------

# Multi-class classification / segmentation
@LOSS_REGISTRY.register("ce")
class CrossEntropyL(TorchLossWrapper):
    def __init__(self, **params):
        super().__init__(nn.CrossEntropyLoss, target_cast=torch.long, **params)


# Binary classification / binary segmentation
@LOSS_REGISTRY.register("bce")
class BCEWithLogitsL(TorchLossWrapper):
    def __init__(self, **params):
        super().__init__(nn.BCEWithLogitsLoss, target_cast=torch.float, **params)


# -------------------------------------------------
# Regression Losses (Keypoint / Heatmap / BBox)
# -------------------------------------------------

# Heatmap regression (keypoint detection)
@LOSS_REGISTRY.register("mse")
class MSEL(TorchLossWrapper):
    def __init__(self, **params):
        super().__init__(nn.MSELoss, **params)


# Coordinate regression (keypoint detection) / bounding box regression
@LOSS_REGISTRY.register("smooth_l1")
class SmoothL1L(TorchLossWrapper):
    def __init__(self, **params):
        super().__init__(nn.SmoothL1Loss, **params)


# -------------------------------------------------
# MONAI Losses (Medical Image Segmentation)
# -------------------------------------------------

# Dice Loss (segmentation)
@LOSS_REGISTRY.register("dice")
class DiceL(TorchLossWrapper):
    def __init__(self, **params):
        super().__init__(DiceLoss, **params)


# Combined Dice + CrossEntropy (segmentation)
@LOSS_REGISTRY.register("dice_ce")
class DiceCEL(TorchLossWrapper):
    def __init__(self, **params):
        super().__init__(DiceCELoss, **params)


# Focal Loss (handles class imbalance) 
@LOSS_REGISTRY.register("focal")
class FocalL(TorchLossWrapper):
    def __init__(self, **params):
        super().__init__(FocalLoss, **params)


# Generalized Dice Loss (handles extreme class imbalance)
@LOSS_REGISTRY.register("generalized_dice")
class GeneralizedDiceL(TorchLossWrapper):
    def __init__(self, **params):
        super().__init__(GeneralizedDiceLoss, **params)


# Tversky Loss (generalization of Dice, useful for medical segmentation)
@LOSS_REGISTRY.register("tversky")
class TverskyL(TorchLossWrapper):
    def __init__(self, **params):
        super().__init__(TverskyLoss, **params)
