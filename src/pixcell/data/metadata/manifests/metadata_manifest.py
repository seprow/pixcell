from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path


@dataclass(frozen=True)
class DicomMetadata:
    """DICOM header metadata extracted from a single slice."""

    # ==========================================================
    # Patient / Study
    # ==========================================================

    patient_id: Optional[str]

    frame_of_reference_uid: Optional[str]

    # =========================================================================
    # Image Geometry (3D Reconstruction, 2.5D Resampling, Registration, Volume)
    # =========================================================================

    rows: Optional[int]
    columns: Optional[int]

    pixel_spacing: Optional[tuple[float, float]]

    slice_thickness: Optional[float]

    image_orientation_patient: Optional[tuple[float, ...]]

    image_position_patient: Optional[tuple[float, float, float]]

    slice_location: Optional[float]

    instance_number: Optional[int]


    # ==========================================================
    # Intensity (HU)
    # ==========================================================

    rescale_intercept: Optional[float]

    rescale_slope: Optional[float]

    window_center: Optional[float]

    window_width: Optional[float]

    # ==========================================================
    # Pixel Encoding (read image correctly)
    # ==========================================================

    bits_allocated: Optional[int]

    bits_stored: Optional[int]

    pixel_representation: Optional[int]

    samples_per_pixel: Optional[int]

    photometric_interpretation: Optional[str]

    image_type: Optional[tuple[str, ...]]

    # ==========================================================
    # Scanner Information (Domain Shift)
    # ==========================================================

    manufacturer: Optional[str]

    manufacturer_model_name: Optional[str]

    software_versions: Optional[str]

    modality: Optional[str]

    # ==========================================================
    # Acquisition Parameters (QC(Quality Control), Domain Shift)
    # ==========================================================

    kvp: Optional[float]

    exposure_time: Optional[float]

    xray_tube_current: Optional[float]

    convolution_kernel: Optional[str]

    protocol_name: Optional[str]

    patient_position: Optional[str]

    reconstruction_diameter: Optional[float]

    gantry_detector_tilt: Optional[float]

    spiral_pitch_factor: Optional[float]

    ctdi_vol: Optional[float]

    # ==========================================================
    # Demographics
    # ==========================================================

    patient_birth_date: Optional[str]

    patient_sex: Optional[str]

    # image geometry
    # Computed later by SeriesMetadataBuilder
    slice_spacing: Optional[float] = None


@dataclass(frozen=True)
class SeriesLevelDicomMetadata:

    series_id: Optional[str] = None

    volume_spacing: Optional[tuple[float, float, float]] = None

    volume_shape: Optional[tuple[int, int, int]] = None

    physical_size_mm: Optional[tuple[float, float, float]] = None

    direction: Optional[tuple[float, ...]] = None

    origin: Optional[tuple[float, float, float]] = None

    is_isotropic: Optional[bool] = None

    slice_thickness: Optional[float] = None



@dataclass(frozen=True)
class AnnotationMetadata:
    """Metadata extracted from a single annotation file."""

    has_segmentation: bool
    has_keypoints: bool
    has_boxes: bool

    # ICH
    segmentation_shape: tuple[int, int] | None
    num_segmentation_classes: int
    class_map: tuple[tuple[str, int], ...]

    # MLS
    keypoint_names: tuple[str, ...]

    # Skull Fracture
    num_boxes: int

@dataclass(frozen=True)
class BoundingBox:

    x: float
    y: float
    width: float
    height: float

@dataclass(frozen=True)
class Keypoint:

    x: float | None
    y: float | None

@dataclass(frozen=True)
class LabelMetadata:
    """
    Slice-level labels extracted from the dataframe.
    """

    # ICH Classification
    any_ich: Optional[int]
    IVH: Optional[int]
    IPH: Optional[int]
    SAH: Optional[int]
    EDH: Optional[int]
    SDH: Optional[int]

    # ICH Area
    IVH_area: Optional[float]
    IPH_area: Optional[float]
    SAH_area: Optional[float]
    EDH_area: Optional[float]
    SDH_area: Optional[float]

    # Skull Fracture
    skull_fracture: Optional[int]

    # Midline Shift
    midline_shift_mm: Optional[float]

    # Triage
    triage_class: Optional[int]

    # AnnotationGeometry
    keypoints: dict[str, Keypoint] | list[dict[str, Keypoint]] | None = None

    bounding_boxes: list[BoundingBox] | list[list[BoundingBox]] | None = None

@dataclass(frozen=True)
class SliceMetadata:
    """
    Complete metadata for a single CT slice.
    """

    # Identity
    patient_id: str
    series_id: str
    sop_uid: str

    # Resources
    dicom_path: Path | None
    annotation_path: Path | None

    # Metadata
    dicom: DicomMetadata
    annotation: AnnotationMetadata | None
    labels: LabelMetadata | None

@dataclass(frozen=True)
class SeriesMetadata:
    """
    Metadata describing a CT series.
    """

    # Identity
    patient_id: str
    series_id: str
    ordered_slice_keys: list[tuple[str, str]]

    dicom: SeriesLevelDicomMetadata
    labels: LabelMetadata

@dataclass
class MetadataManifest:
    """
    Root metadata artifact produced by the metadata extraction stage.
    """

    # key = series_id
    series: dict[str, SeriesMetadata] = field(default_factory=dict)

    # key = (series_id, sop_uid)
    slices: dict[tuple[str, str], SliceMetadata] = field(default_factory=dict)