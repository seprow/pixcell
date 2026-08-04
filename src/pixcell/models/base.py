from __future__ import annotations

import torch.nn as nn


class BaseModel(nn.Module):
    """
    Base class for all PixCell models.

    Child classes should implement:

        - forward()
        - parameter_groups() (optional)

    parameter_groups() is used by OptimizerBuilder to create optimizer
    parameter groups with different hyperparameters.
    """

    def __init__(self):
        super().__init__()

    def forward(self, *args, **kwargs):
        """
            self.encoder = ...
            self.decoder = ...
            self.seg_head = ...
            self.cls_head = ...
        """
        raise NotImplementedError

    def parameter_groups(self) -> dict[str, object]:
        """
        Returns parameter groups for the optimizer.

        Returns:
            Dictionary mapping group names to parameter iterables.

            return {
                "encoder": self.encoder.parameters(),
                "decoder": self.decoder.parameters(),
                "seg_head": self.seg_head.parameters(),
                "cls_head": self.cls_head.parameters(),
            }

        Default:
            {"default": self.parameters()}
        """
        return {
            "default": self.parameters()
        }