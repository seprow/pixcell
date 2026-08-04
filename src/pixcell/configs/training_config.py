from dataclasses import dataclass, field
from typing import Any, Optional

"""
config.training.optimizer :
    optimizer: AdamW
    params:
        weight_decay: 0.01
        lr: 1e-4

    parameter_groups:
        - name: encoder
        params:
            lr: 1e-5

        - name: decoder
        params:
            lr: 1e-4

        - name: seg_head
        params:
            lr: 5e-4

"""

@dataclass
class ParameterGroupConfig:
    name: str
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class OptimizerConfig:
    optimizer: str = "AdamW"
    params: dict[str, Any] = field(default_factory=dict)
    parameter_groups: list[ParameterGroupConfig] = field(default_factory=list)


@dataclass
class SchedulerConfig:
    name: str
    params: dict[str, Any] = field(default_factory=dict)
    step_per_batch: bool = True


@dataclass
class LossItemConfig:
    name: str
    weight: float = 1.0
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class TargetLossConfig:
    target: str
    losses: list[LossItemConfig] = field(default_factory=list)

    strategy: Optional[str] = None # weighted_sum, 
    strategy_params: dict[str, Any] = field(default_factory=dict)


@dataclass
class LossConfig:
    targets: list[TargetLossConfig] = field(default_factory=list)

    strategy: Optional[str] = None # sum,
    strategy_params: dict[str, Any] = field(default_factory=dict)


@dataclass 
class MetricItemConfig:
    name: str
    params: dict[str, Any] = field(default_factory=dict)
    postprocessor: Optional[str] = None


@dataclass
class TargetMetricConfig:
    target: str
    metrics: list[MetricItemConfig] = field(default_factory=list)


@dataclass
class TargetConfig:
    target_name: str # pixcell.data.dataloader.target_filter





