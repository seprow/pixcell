import torch.nn as nn


class TargetLoss(nn.Module):
    """
    Computes the configured losses for a single target.
    """

    def __init__(
        self,
        target,
        losses,
        strategy=None,
        strategy_params=None,
    ):
        super().__init__()

        self.target = target

        self.losses = nn.ModuleDict({
            name: fn
            for name, fn, _ in losses
        })

        self.weights = {
            name: weight
            for name, _, weight in losses
        }

        self.strategy = strategy
        self.strategy_params = strategy_params or {}

    def forward(
        self,
        outputs,
        targets,
    ):

        prediction = outputs[self.target]
        target = targets[self.target]

        loss_values = {}
        logs = {}

        for name, loss_fn in self.losses.items():

            loss = loss_fn(
                prediction,
                target,
            )

            loss_values[name] = loss

            logs[f"{self.target}/{name}"] = loss.detach()

        # only one loss

        if len(loss_values) == 1:

            total = next(iter(loss_values.values()))

            return total, logs

        # multiple losses

        if self.strategy is None:

            raise ValueError(
                f"Target '{self.target}' has multiple losses but no strategy was specified."
            )

        if self.strategy == "weighted_sum":

            total = 0.0

            for name, loss in loss_values.items():

                total += self.weights[name] * loss

            return total, logs

        raise ValueError(
            f"Unknown target loss strategy: {self.strategy}"
        )