import torch

from pixcell.configs import Config


class OptimizerBuilder:

    @staticmethod
    def build(model, config: Config):

        optimizer_cfg = config.training.optimizer

        # Parameter groups 

        if optimizer_cfg.parameter_groups:

            available_groups = model.parameter_groups()

            param_groups = []

            for group_cfg in optimizer_cfg.parameter_groups:

                if group_cfg.name not in available_groups:
                    raise ValueError(
                        f"Unknown parameter group '{group_cfg.name}'. "
                        f"Available groups: {list(available_groups.keys())}"
                    )

                param_groups.append({
                    "params": available_groups[group_cfg.name],
                    **group_cfg.params,
                })

        else:

            param_groups = model.parameters()

        # Optimizer 

        name = optimizer_cfg.optimizer.lower()

        if name == "adam":

            optimizer = torch.optim.Adam(
                param_groups,
                **optimizer_cfg.params,
            )

        elif name == "adamw":

            optimizer = torch.optim.AdamW(
                param_groups,
                **optimizer_cfg.params,
            )

        elif name == "sgd":

            optimizer = torch.optim.SGD(
                param_groups,
                **optimizer_cfg.params,
            )

        elif name == "rmsprop":

            optimizer = torch.optim.RMSprop(
                param_groups,
                **optimizer_cfg.params,
            )

        elif name == "adagrad":

            optimizer = torch.optim.Adagrad(
                param_groups,
                **optimizer_cfg.params,
            )

        else:

            raise ValueError(
                f"Unknown optimizer '{optimizer_cfg.optimizer}'."
            )

        return optimizer