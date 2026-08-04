from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional


@dataclass(frozen=True)
class SliceManifest:
    """Represents a single slice in the dataset."""

    series_id: str
    sop_uid: str
    patient_id: str 

    dicom_path: Optional[Path]
    annotation_path: Optional[Path]

    dataframe_index: Optional[int]

    has_dicom: bool
    has_annotation: bool
    has_dataframe: bool

@dataclass(frozen=True)
class SeriesManifest:
    """Represents a CT series."""

    series_id: str

    dicom_directory: Optional[Path]
    annotation_directory: Optional[Path]

    slice_keys: list[tuple[str, str]]

    num_dicoms: int
    num_annotations: int

@dataclass(frozen=True)
class PatientManifest:
    """Represents a patient."""

    patient_id: str

    series_ids: list[str]

@dataclass(frozen=True)
class DatasetSummary:

    num_patients: int
    num_series: int
    num_slices: int

    annotated_series: int
    annotated_slices: int

    missing_dicoms: int
    missing_annotations: int
    missing_dataframe_rows: int

@dataclass(frozen=True)
class DatasetStructure:
    dataset_root: Path

    train_dir: Path
    annotation_dir: Optional[Path]
    dataframe_path: Optional[Path]

    is_valid: bool = True
    missing_parts: tuple[str, ...] = ()

@dataclass(frozen=True)
class DatasetManifest:
    """Root manifest produced by the discovery stage."""

    name: str
    structure: DatasetStructure | None = None

    patients: dict[str, PatientManifest] = field(default_factory=dict)
    series: dict[str, SeriesManifest] = field(default_factory=dict)
    # key = (series_id, sop_uid)
    slices: dict[tuple[str, str], SliceManifest] = field(default_factory=dict)

    summary: DatasetSummary | None = None

    def filter_slices(
        self,
        *,
        dicom: bool | None = None,
        annotation: bool | None = None,
        dataframe: bool | None = None,
    ) -> dict[tuple[str, str], SliceManifest]:
        return {
            key: slice_manifest
            for key, slice_manifest in self.slices.items()
            if (
                (dicom is None or slice_manifest.has_dicom == dicom)
                and (annotation is None or slice_manifest.has_annotation == annotation)
                and (dataframe is None or slice_manifest.has_dataframe == dataframe)
            )
        }