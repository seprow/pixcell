from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from pixcell.utils.io import read_json_file
from pixcell.utils.registry import ANNOTATION_BUILDER_REGISTRY
from pixcell.data.engine.manifests import SliceAnnotation, VolumeAnnotation
from pixcell.data.engine.reconstruction.annotation.base import BaseAnnotationBuilder


@ANNOTATION_BUILDER_REGISTRY.register("cv2_guassian_heatmap")
class SliceHeatmapBuilder(BaseAnnotationBuilder):
    """
    Build a keypoint heatmap from an annotation file.
    """

    def __init__(
        self,
        sigma: float = 5.0,
    ):
        self._sigma = sigma

    def build(
        self,
        source: Path,
    ) -> SliceAnnotation:

        if not isinstance(source, Path):
            raise TypeError(
                "SliceHeatmapBuilder expects a Path."
            )

        annotation = read_json_file(source)

        shape = tuple(annotation["segmentation_rle"]["shape"])

        heatmap = np.zeros(shape, dtype=np.float32)

        for point in annotation["keypoints"].values():

            if point is None:
                continue

            x, y = point

            cv2.circle(
                heatmap,
                center=(x, y),
                radius=0,
                color=1.0,
                thickness=-1,
            )

        heatmap = cv2.GaussianBlur(
            heatmap,
            ksize=(0, 0),
            sigmaX=self._sigma,
            sigmaY=self._sigma,
        )

        if heatmap.max() > 0:
            heatmap /= heatmap.max()

        return SliceAnnotation(
            annotation=heatmap,
            slice_id=(
                source.parent.name,
                source.stem,
            ),
        )
    
@ANNOTATION_BUILDER_REGISTRY.register("cv2_guassian_heatmap_volume")
class VolumeHeatmapBuilder(BaseAnnotationBuilder):
    """
    Build volume heatmaps from annotation files.
    """

    def __init__(
        self,
        sigma: float = 5.0,
    ):
        self._sigma = sigma

    def build(
        self,
        source: list[Path],
    ) -> VolumeAnnotation:

        if not isinstance(source, list):
            raise TypeError(
                "VolumeHeatmapBuilder expects a list of Paths."
            )

        if not source:
            raise ValueError(
                "Source list cannot be empty."
            )

        heatmaps = []
        ordered_slice_ids = []

        for annotation_path in source:

            annotation = read_json_file(annotation_path)

            shape = tuple(
                annotation["segmentation_rle"]["shape"]
            )

            heatmap = np.zeros(
                shape,
                dtype=np.float32,
            )

            for point in annotation["keypoints"].values():

                if point is None:
                    continue

                x, y = point

                cv2.circle(
                    heatmap,
                    center=(x, y),
                    radius=0,
                    color=1.0,
                    thickness=-1,
                )

            heatmap = cv2.GaussianBlur(
                heatmap,
                ksize=(0, 0),
                sigmaX=self._sigma,
                sigmaY=self._sigma,
            )

            if heatmap.max() > 0:
                heatmap /= heatmap.max()

            heatmaps.append(heatmap)

            ordered_slice_ids.append(
                (
                    annotation_path.parent.name,
                    annotation_path.stem,
                )
            )

        return VolumeAnnotation(
            annotation=np.stack(
                heatmaps,
                axis=0,
            ),
            ordered_slice_ids=ordered_slice_ids,
        )

def _gaussian_heatmap(
        self,
        shape: tuple[int, int],
        center: tuple[int, int],
    ) -> np.ndarray:

        h, w = shape

        y = np.arange(h, dtype=np.float32)
        x = np.arange(w, dtype=np.float32)

        yy, xx = np.meshgrid(y, x, indexing="ij")

        cy, cx = center

        heatmap = np.exp(
            -(
                (xx - cx) ** 2 +
                (yy - cy) ** 2
            ) / (2 * self.sigma ** 2)
        )

        return heatmap.astype(np.float32)
