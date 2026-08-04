from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    EnsureTyped,
    RandCropByLabelClassesd,
    NormalizeIntensityd,
    RandScaleIntensityd,
    RandShiftIntensityd,
    RandFlipd,
    RandRotate90d,
)

from pixcell.utils.registry import TRANSFORM_PIPELINE

from pixcell.data.engine.transforms import(
    ConvertToMultiChannelICHd,
    Windowingd
)


@TRANSFORM_PIPELINE.register("default")
def default_transform():

    return Compose([
        RandFlipd(
            keys=["image", "ich_multiclass_segmentation_series"],
            prob=0.5,
        ),
        RandRotate90d(
            keys=["image", "ich_multiclass_segmentation_series"],
            prob=0.5,
        ),
    ])





