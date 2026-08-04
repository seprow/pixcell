from __future__ import annotations
from pathlib import Path
import numpy as np

from pixcell.utils import read_json_file
from pixcell.utils.registry import ANNOTATION_BUILDER_REGISTRY
from pixcell.data.engine.manifests import (
    SliceAnnotation, 
    VolumeAnnotation
)
from ..rle_codec import SegmentationData
from ...base import BaseAnnotationBuilder


@ANNOTATION_BUILDER_REGISTRY.register(
    "binary_mask_any_ich",
)
class BinaryMaskAnyICHBuilder(
    BaseAnnotationBuilder,
):

    def build(
        self,
        source: Path | list[Path],
    ) -> SliceAnnotation:

        annotation = read_json_file(
            source
        )

        segmentation = (
            SegmentationData.from_annotation(
                annotation
            )
        )

        return SliceAnnotation(
            annotation=segmentation.binary_mask_any_ich,
            slice_id=(
                source.parent.name,
                source.stem,
            ),
        )
    

@ANNOTATION_BUILDER_REGISTRY.register(
    "binary_mask_any_ich_volume",
)
class BinaryMaskAnyICHVolumeBuilder(
    BaseAnnotationBuilder,
):

    def build(
        self,
        source: Path | list[Path],
    ) -> VolumeAnnotation:

        masks = []

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

            masks.append(
                segmentation.binary_mask_any_ich
            )

            ordered_slice_ids.append(
                (
                    path.parent.name,
                    path.stem,
                )
            )

        return VolumeAnnotation(
            annotation=np.stack(
                masks,
                axis=0,
            ),
            ordered_slice_ids=ordered_slice_ids,
        )