import torch
import torch.nn as nn


class BaseLoss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, pred, target, **kwargs):
        raise NotImplementedError


class TorchLossWrapper(BaseLoss):

    def __init__(self, loss_fn, target_cast=None, **params):
        super().__init__()
        self.loss = loss_fn(**params)
        self.target_cast = target_cast

    def forward(self, pred, target, **kwargs):

        if self.target_cast is not None:
            target = target.to(
                dtype=self.target_cast
            )

        return self.loss(pred, target, **kwargs)
