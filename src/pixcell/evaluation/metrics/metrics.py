from monai.metrics import (
    DiceMetric,
    MeanIoU,
    HausdorffDistanceMetric,
    SurfaceDistanceMetric
)

from torchmetrics.classification import (
    BinaryRecall,
    BinaryPrecision,
    BinaryF1Score,
    BinaryAveragePrecision,
    F1Score,
    AUROC,
    Precision,
    Recall,
    Specificity,
    ConfusionMatrix,
)

from torchmetrics.regression import (
    MeanAbsoluteError,
    MeanSquaredError,
)

from pixcell.utils.registry import METRIC_REGISTRY


@METRIC_REGISTRY.register("dice")
def build_dice(**kwargs):
    return DiceMetric(
        **kwargs,
    )


@METRIC_REGISTRY.register("iou")
def build_iou(**kwargs):
    return MeanIoU(
        include_background=True,
        reduction="mean",
        **kwargs,
    )


@METRIC_REGISTRY.register("hausdorff95")
def build_hd95(**kwargs):
    return HausdorffDistanceMetric(
        percentile=95,
        include_background=True,
        **kwargs,
    )


@METRIC_REGISTRY.register("surface_distance")
def build_surface_distance(**kwargs):
    return SurfaceDistanceMetric(
        include_background=True,
        **kwargs,
    )


@METRIC_REGISTRY.register("binary_recall")
def build_binary_recall(**kwargs):
    return BinaryRecall(**kwargs)


@METRIC_REGISTRY.register("binary_precision")
def build_binary_precision(**kwargs):
    return BinaryPrecision(**kwargs)


@METRIC_REGISTRY.register("binary_f1")
def build_binary_f1(**kwargs):
    return BinaryF1Score(**kwargs)


@METRIC_REGISTRY.register("binary_pr_auc")
def build_binary_pr_auc(**kwargs):
    return BinaryAveragePrecision(**kwargs)


@METRIC_REGISTRY.register("f1")
def build_f1(task, num_classes, **kwargs):
    return F1Score(
        task=task,
        num_classes=num_classes,
        **kwargs,
    )


@METRIC_REGISTRY.register("precision")
def build_precision(task, num_classes, **kwargs):
    return Precision(
        task=task,
        num_classes=num_classes,
        **kwargs,
    )


@METRIC_REGISTRY.register("recall")
def build_recall(task, num_classes, **kwargs):
    return Recall(
        task=task,
        num_classes=num_classes,
        **kwargs,
    )


@METRIC_REGISTRY.register("mae")
def build_mae(**kwargs):
    return MeanAbsoluteError(**kwargs)


@METRIC_REGISTRY.register("rmse")
def build_rmse(**kwargs):
    return MeanSquaredError(
        squared=False,
        **kwargs,
    )