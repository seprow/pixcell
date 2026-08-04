from __future__ import annotations
from pathlib import Path
import pandas as pd

from pixcell.data.metadata.manifests import LabelMetadata, Keypoint, BoundingBox 
from pixcell.utils import read_json_file

class LabelExtractor:
    """
    Extract slice labels from the dataframe.
    """

    def __init__(self, dataframe_path: Path):
        self._df = pd.read_pickle(dataframe_path)

    def extract(
        self,
        dataframe_index: int | None,
        annotation_path: Path | None
    ) -> LabelMetadata | None:

        if dataframe_index is None:
            return None

        row = self._df.iloc[dataframe_index]
        
        keypoints = None
        bounding_boxes = None

        if annotation_path is not None:
            keypoints, bounding_boxes = (
                self._load_annotation(annotation_path)
            )

        return LabelMetadata(

            any_ich=self._to_binary(
                row["AnyICH"]
            ),

            IVH=self._to_binary(
                row["IntraventricularHemorrhage"]
            ),

            IPH=self._to_binary(
                row["IntraparenchymalHemorrhage"]
            ),

            SAH=self._to_binary(
                row["SubarachnoidHemorrhage"]
            ),

            EDH=self._to_binary(
                row["EpiduralHemorrhage"]
            ),

            SDH=self._to_binary(
                row["SubduralHemorrhage"]
            ),

            IVH_area=self._to_float(
                row["IntraventricularHemorrhage_Area"]
            ),

            IPH_area=self._to_float(
                row["IntraparenchymalHemorrhage_Area"]
            ),

            SAH_area=self._to_float(
                row["SubarachnoidHemorrhage_Area"]
            ),

            EDH_area=self._to_float(
                row["EpiduralHemorrhage_Area"]
            ),

            SDH_area=self._to_float(
                row["SubduralHemorrhage_Area"]
            ),

            skull_fracture=self._to_binary(
                row["SkullFracture"]
            ),

            midline_shift_mm=self._to_float(
                row["MidlineShiftMM"]
            ),

            triage_class=self._to_int(
                row["triage_class"]
            ),

            # Geometry
            keypoints=keypoints,
            bounding_boxes=bounding_boxes,
        )

    @staticmethod
    def _load_annotation(
        annotation_path: Path,
    ) -> tuple[
        dict[str, Keypoint] | None,
        list[BoundingBox] | None,
    ]:

        data = read_json_file(annotation_path)

        keypoints = None

        keypoints = {
            name: (
                Keypoint(x=point[0], y=point[1])
                if point is not None
                else Keypoint(x=None, y=None)
            )
            for name, point in data["keypoints"].items()
        }

        bounding_boxes = None

        if "boxes_xywh" in data:
            bounding_boxes = [
                BoundingBox(
                    x=box[0],
                    y=box[1],
                    width=box[2],
                    height=box[3],
                )
                for box in data["boxes_xywh"]
            ]

        return keypoints, bounding_boxes

    @staticmethod
    def _to_binary(value):

        if pd.isna(value):
            return None

        return int(value)

    @staticmethod
    def _to_float(value):

        if pd.isna(value):
            return None

        return float(value)
    
    @staticmethod
    def _to_int(value) -> int | None:

        if pd.isna(value):
            return None

        return int(value) # I Know. 
    
        
