import torch
from monai.metrics import CumulativeIterationMetric

from pixcell.evaluation.metrics import (
    TorchMetricWrapper,
    MonaiMetricWrapper
)
from pixcell.utils.registry import (
    METRIC_REGISTRY,
    POSTPROCESSOR_REGISTRY,
)


class MetricBuilder:

    def __init__(self, config, device):

        self.device = device 

        self.target_metrics = {}

        for target_cfg in config.training.metrics:

            metrics = {}

            for metric_cfg in target_cfg.metrics:

                metric = self._build_metric(metric_cfg)

                metrics[metric_cfg.name] = metric

            self.target_metrics[target_cfg.target] = metrics


    def _build_metric(self, metric_cfg):

        metric = METRIC_REGISTRY.build(
            metric_cfg.name,
            **metric_cfg.params,
        )

        metric = metric

        postprocessor = None

        if metric_cfg.postprocessor is not None:
            postprocessor = POSTPROCESSOR_REGISTRY.build(
                metric_cfg.postprocessor
            )


        if isinstance(metric, CumulativeIterationMetric):

            return MonaiMetricWrapper(
                metric=metric,
                postprocessor=postprocessor,
            )

        return TorchMetricWrapper(
            metric=metric,
            postprocessor=postprocessor,
        )


    def reset(self):

        for metrics in self.target_metrics.values():
            for m in metrics.values():
                m.reset()


    def update(self, outputs, targets):

        for target_name, metrics in self.target_metrics.items():

            preds = outputs[target_name]
            tg = targets[target_name]

            if preds is None:
                raise ValueError(f"Missing output for target '{target_name}'")

            if tg is None:
                raise ValueError(f"Missing target '{target_name}'")

            for metric in metrics.values():
                metric.update(preds, tg)


    def compute(self):

        results = {}

        for target_name, metrics in self.target_metrics.items():

            for metric_name, metric in metrics.items():

                val = metric.compute()

                if isinstance(val, torch.Tensor):

                    if val.numel() == 1:
                        val = val.item()
                    else:
                        val = val.cpu().tolist()

                results[f"{target_name}/{metric_name}"] = val

        return results


