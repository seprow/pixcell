from __future__ import annotations
from pathlib import Path
import numpy as np

from pixcell.utils import read_json_file
from pixcell.utils.registry import ANNOTATION_BUILDER_REGISTRY
from pixcell.data.engine.manifests import (
    SliceAnnotations, 
    VolumeAnnotations
)
from ..rle_codec import SegmentationData
from ...base import BaseAnnotationBuilder


@ANNOTATION_BUILDER_REGISTRY.register(
    "binary_masks_by_class",
)
class BinaryMasksByClassBuilder(
    BaseAnnotationBuilder,
):

    def build(
        self,
        source: Path | list[Path],
    ) -> SliceAnnotations:

        annotation = read_json_file(
            source
        )

        segmentation = (
            SegmentationData.from_annotation(
                annotation
            )
        )

        return SliceAnnotations(
            annotation=segmentation.binary_masks_by_class,
            slice_id=(
                source.parent.name,
                source.stem,
            ),
        )


@ANNOTATION_BUILDER_REGISTRY.register(
    "binary_masks_by_class_volume",
)
class BinaryMasksByClassVolumeBuilder(
    BaseAnnotationBuilder,
):

    def build(
        self,
        source: Path | list[Path],
    ) -> VolumeAnnotations:

        annotations = {}

        ordered_slice_ids = []

        for path in source:

            annotation = read_json_file(
                path
            )

            segmentation = (
                SegmentationData.from_annotation(
                    annotation
                )
            )

            masks = (
                segmentation.binary_masks_by_class
            )

            if not annotations:
                annotations = {
                    key: []
                    for key in masks
                }

            for key, mask in masks.items():
                annotations[key].append(
                    mask
                )

            ordered_slice_ids.append(
                (
                    path.parent.name,
                    path.stem,
                )
            )

        annotations = {
            key: np.stack(
                value,
                axis=0,
            )
            for key, value in annotations.items()
        }

        return VolumeAnnotations(
            annotation=annotations,
            ordered_slice_ids=ordered_slice_ids,
        )
    


