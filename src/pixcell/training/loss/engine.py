import torch.nn as nn
import torch

class LossEngine(nn.Module):
    """
    Computes and combines losses across all configured targets.
    """

    def __init__(
        self,
        target_losses,
        strategy=None,
        strategy_params=None,
    ):
        super().__init__()

        self.target_losses = nn.ModuleList(target_losses)

        self.strategy = strategy
        self.strategy_params = strategy_params or {}

    def forward(
        self,
        outputs,
        targets,
    ):

        target_loss_values = []

        logs = {}

        for target_loss in self.target_losses:

            loss, target_logs = target_loss(
                outputs,
                targets,
            )

            target_loss_values.append(loss)

            logs.update(target_logs)

    
        # Only one target loss

        if len(target_loss_values) == 1:

            total = target_loss_values[0]

            logs["total"] = total.detach()

            return total, logs


        # Multiple target losses

        if self.strategy is None:

            raise ValueError(
                "A strategy must be specified when multiple target losses are configured."
            )

        if self.strategy == "sum":

            total = torch.stack(target_loss_values).sum()

            logs["total"] = total.detach()

            return total, logs

        raise ValueError(
            f"Unknown loss strategy: {self.strategy}"
        )