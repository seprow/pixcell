from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from pathlib import Path


@dataclass
class ImageConfig:
    """
    reader:
        - dicom_volume_reader
        - dicom_slice_reader
    """
    reader: str = "dicom_volume_reader"

@dataclass
class AnnotationConfig:
    """
    Prediction target.

    task:
        - ich
        - mls
        - skull_fracture

    builder:

        ICH
            slice_wise
                - multiclass_mask
                - multiclass_mask_for_classes
                - binary_mask_any_ich
                - binary_mask_for_class
                - binary_masks_by_class (builder only)

            series_wise
                - multiclass_mask_volume
                - multiclass_mask_for_classes_volume
                - binary_mask_any_ich_volume
                - binary_mask_for_class_volume
                - binary_masks_by_class_volume (builder)

        MLS
            slice_wise
                - cv2_guassian_heatmap

            series_wise
                - cv2_guassian_heatmap_volume

        Skull Fracture


    """

    task: str

    builder: str

    parameters: dict[str, Any] = field(
        default_factory=dict
    )

    cache_key: Optional[str] = None

@dataclass
class AnnotationsConfig:
    """
    Ground-truth targets for training.

    Multiple targets enable multi-task learning.
    """

    targets: list[AnnotationConfig] = field(
        default_factory=list
    )

@dataclass
class ResampleConfig:
    """
    interpolator:
        - nearest
        - linear
        - bspline
    """
    spacing: Optional[tuple[float, float, float]]  = None

    interpolator: Optional[str] = None

    default_pixel_value: Optional[int] = None

@dataclass
class ResamplesConfig:
    """resampling."""

    enabled: bool = False

    pipeline: str = "image_spacing_resampler"

    image: ResampleConfig = field(
        default_factory=lambda: ResampleConfig(
            spacing=(1, 1, 1),
            interpolator="bspline",
            default_pixel_value=-1000
        )
    )

    annotation: ResampleConfig = field(
        default_factory=lambda: ResampleConfig(
            spacing=(1, 1, 1),
            interpolator="nearest",
            default_pixel_value=0
        )
    )  

@dataclass
class ReconstructionConfig:
    """
    Image and target reconstruction.

    """
    enabled: bool = True

    cache: bool = True

    load: bool = False

    image: ImageConfig = field(
        default_factory=ImageConfig
    )

    annotation: AnnotationsConfig = field(
        default_factory=AnnotationsConfig
    )

    resample: ResamplesConfig = field(
        default_factory=ResamplesConfig
    )


