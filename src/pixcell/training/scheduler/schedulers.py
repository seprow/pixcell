import torch
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    ReduceLROnPlateau
)

from pixcell.training.scheduler import TorchSchedulerWrapper, WarmupDecayScheduler
from pixcell.utils.registry import LR_SCHEDULERS
 

@LR_SCHEDULERS.register("warmup") # batch
class WarmupScheduler(TorchSchedulerWrapper):

    def __init__(self, optimizer, **params):
        super().__init__(WarmupDecayScheduler, optimizer, **params)


@LR_SCHEDULERS.register("cosine")
class CosineAnnealingScheduler(TorchSchedulerWrapper):

    def __init__(self, optimizer, **params):
        super().__init__(CosineAnnealingLR, optimizer, **params)


@LR_SCHEDULERS.register("reduce_on_plateau") # epoch
class ReduceOnPlateauScheduler(TorchSchedulerWrapper):

    def __init__(self, optimizer, **params):
        super().__init__(ReduceLROnPlateau, optimizer, **params)








