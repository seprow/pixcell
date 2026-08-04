from __future__ import annotations

import json
from pathlib import Path

from pixcell.data.metadata.manifests import AnnotationMetadata
from pixcell.utils import read_json_file


class AnnotationMetadataExtractor:
    """Extract metadata from an annotation JSON file."""

    def extract(
        self,
        annotation_path: Path | None,
    ) -> AnnotationMetadata | None:

        if annotation_path is None:
            return None

        annotation = read_json_file(annotation_path)

        segmentation = annotation.get("segmentation_rle")
        classmap = annotation.get("class_map", [])
        keypoints = annotation.get("keypoints", {})
        boxes = annotation.get("boxes_xywh", [])

        has_segmentation = self._has_segmentation(annotation, segmentation)
        has_keypoints = self._has_keypoints(annotation)
        has_boxes = self._has_boxes(annotation)

        return AnnotationMetadata(
            # File existence
            has_segmentation=has_segmentation,
            has_keypoints=has_keypoints,
            has_boxes=has_boxes,

            # Segmentation
            segmentation_shape=(
                tuple(segmentation["shape"])
                if has_segmentation
                else None
            ),

            num_segmentation_classes=len(classmap), 

            class_map=tuple(
                tuple([cls["name"], cls["value"]])
                for cls in classmap
            ),

            # Keypoints
            keypoint_names=tuple(keypoints.keys()) if has_keypoints else None, 

            # Bounding Boxes
            num_boxes=len(boxes) if has_boxes else None, 
        )
    
    def _has_segmentation(self,annotation, segmentation):
  
        return (
            "segmentation_rle" in annotation
            and isinstance(segmentation, dict)
            and "shape" in segmentation
            and "counts" in segmentation
        )

    def _has_keypoints(self, annotation):
        return "keypoints" in annotation

    def _has_boxes(self, annotation):
        return "boxes_xywh" in annotation
    