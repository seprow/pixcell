from pixcell.utils.registry import LOSS_REGISTRY
from pixcell.configs import Config

from .target_loss import TargetLoss
from .engine import LossEngine


def build_loss(config: Config):
    cfg = config.training.loss

    target_losses = []

    for target_cfg in cfg.targets:

        losses = []

        for item in target_cfg.losses:

            loss_fn = LOSS_REGISTRY.build(
                item.name,
                **item.params,
            )

            losses.append(
                (
                    item.name,
                    loss_fn,
                    item.weight,
                )
            )

        target_losses.append(

            TargetLoss(
                target=target_cfg.target,
                losses=losses,
                strategy=target_cfg.strategy,
                strategy_params=target_cfg.strategy_params,
            )

        )

    return LossEngine(
        target_losses=target_losses,
        strategy=cfg.strategy,
        strategy_params=cfg.strategy_params,
    )