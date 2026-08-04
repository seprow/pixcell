from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from pixcell.data.dataset_discovery.manifests import DatasetStructure


class DatasetRootScanner:
    """discover dataset top-level structure"""

    def __init__(
        self,
        dataset_root: Path,
        train_dir_name: str = "training",
        annotation_dir_name: str  = "annotations",
        dataframe_name: str = "training_df.pkl",
    ):
        self.dataset_root = dataset_root
        self.train_dir_name = train_dir_name
        self.annotation_dir_name = annotation_dir_name
        self.dataframe_name = dataframe_name

    # public API
    def scan(self) -> DatasetStructure:
        train_dir = self._find_optional_dir_or_file(self.train_dir_name)

        annotation_dir = self._find_optional_dir_or_file(self.annotation_dir_name)

        dataframe_path = self._find_optional_dir_or_file(self.dataframe_name)

        missing_parts = self._validate(train_dir, annotation_dir, dataframe_path)

        return DatasetStructure(
            dataset_root=self.dataset_root,
            train_dir=train_dir,
            annotation_dir=annotation_dir,
            dataframe_path=dataframe_path,
            is_valid=len(missing_parts) == 0,
            missing_parts=tuple(missing_parts),
        )

    # filesystem-only
    def _find_optional_dir_or_file(self, name: str) -> Optional[Path]:

        path = self.dataset_root / name
        return path if path.exists() else None


    def _validate(
        self,
        train_dir: Path,
        annotation_dir: Optional[Path],
        dataframe_path: Optional[Path],
    ) -> list[str]:

        missing = []

        if train_dir is None:
            missing.append("train_dir")

        if annotation_dir is None:
            missing.append("annotation_dir")

        if dataframe_path is None:
            missing.append("dataframe")

        return missing
    
    