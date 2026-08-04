from __future__ import annotations
from pathlib import Path

from pixcell.configs import Config
from pixcell.utils import PathResolver
from pixcell.data.metadata.manifests import (
    SeriesMetadata, 
    SliceMetadata, 
    SeriesLevelDicomMetadata, 
    LabelMetadata
)
from pixcell.data.dataset_discovery.manifests import SeriesManifest

from pixcell.data.engine.reconstruction.image import (
    DicomVolumeReader,
    DicomSeriesOrder,
    DicomPathMatcher
)

class SeriesMetadataBuilder:

    def __init__(self, config: Config):
        self._config = config
        self._path_resolver = PathResolver(self._config)
        self._reader = DicomVolumeReader()
        self._order = DicomSeriesOrder()
        self._match = DicomPathMatcher()

    def _build_dicom_metadata(
        self,
        series_manifest: SeriesManifest,
        slice_metadata: list[SliceMetadata],
    )-> tuple[
        SeriesLevelDicomMetadata,
        list[tuple[str, str]],
    ]:
        
        if not slice_metadata:
            raise ValueError("Series contains no slices.")

        first = slice_metadata[0]

        reference = [self._path_resolver.slice_path(id, uid) 
                     for id, uid in series_manifest.slice_keys]

        _ordered_dicom_paths = self._order.get_ordered_paths(
            series_manifest.dicom_directory
        )

        _matched_dicom_paths = self._match.match(
            _ordered_dicom_paths, reference
        )
        
        volume = self._reader.read(
            _matched_dicom_paths
           
        )

        image = volume.image

        size = image.GetSize()
        spacing = image.GetSpacing()

        physical_size = tuple(
            s * sp
            for s, sp in zip(size, spacing)
        )

        is_isotropic = (
            abs(spacing[0] - spacing[1]) < 1e-3
            and abs(spacing[1] - spacing[2]) < 1e-3
        )

        return SeriesLevelDicomMetadata(
            series_id=series_manifest.series_id,
            volume_spacing=spacing,
            slice_thickness=first.dicom.slice_thickness,
            volume_shape=size,
            physical_size_mm=physical_size,
            direction=image.GetDirection(),
            origin=image.GetOrigin(),
            is_isotropic=is_isotropic,
        ), volume.ordered_slice_ids
        
    def _build_label(
        self,
        series_manifest: SeriesManifest,
        slice_metadata: list[SliceMetadata],
        dicom_metadata: SeriesLevelDicomMetadata,
    ) -> LabelMetadata | None:

        labels = [
            s.labels
            for s in slice_metadata
            if s.labels is not None
        ]

        if not labels:
            return None

        def _values(attr: str) -> list:
            values = []

            for label in labels:
                value = getattr(label, attr)
                if value is not None:
                    values.append(value)

            return values

        def _max(attr: str):
            values = _values(attr)
            return max(values) if values else None

        def _sum(attr: str):
            values = _values(attr)
            return sum(values) if values else None

        def _unique(attr: str):
            values = set(_values(attr))

            if not values:
                return None

            if len(values) > 1:
                raise ValueError(
                    f"Inconsistent '{attr}' values within series: {values}"
                )

            return values.pop()

        def _mm_to_ml3(attr: str):
            areas_mm = _values(attr)
            volume_spacing = dicom_metadata.volume_spacing

            voxel_volume_mm3 = volume_spacing[0] * volume_spacing[1] * volume_spacing[2]
            total_mm3 = sum(area * voxel_volume_mm3 for area in areas_mm)
    
            total_ml3 = total_mm3 / 1000

            return total_ml3


        keypoints = [
            label.keypoints
            for label in labels
            if label.keypoints is not None
        ]

        bounding_boxes = [
            label.bounding_boxes
            for label in labels
            if label.bounding_boxes is not None
        ]

        return LabelMetadata(

            # ICH Classification 
            any_ich=_max("any_ich"),
            IVH=_max("IVH"),
            IPH=_max("IPH"),
            SAH=_max("SAH"),
            EDH=_max("EDH"),
            SDH=_max("SDH"),

            # ICH Area 
            IVH_area=_mm_to_ml3("IVH_area"),
            IPH_area=_mm_to_ml3("IPH_area"),
            SAH_area=_mm_to_ml3("SAH_area"),
            EDH_area=_mm_to_ml3("EDH_area"),
            SDH_area=_mm_to_ml3("SDH_area"),

            # Skull Fracture 
            skull_fracture=_max("skull_fracture"),

            # Midline Shift 
            midline_shift_mm=_max("midline_shift_mm"),

            # Triage 
            triage_class=_unique("triage_class"),

            # Geometry
            keypoints=keypoints or None,
            bounding_boxes=bounding_boxes or None,
        )
        
    
    def build(
        self,
        series_manifest: SeriesManifest,
        slice_metadata: list[SliceMetadata],
    ) -> SeriesMetadata:

        dicom_metadata, ordered_slice_keys = (
            self._build_dicom_metadata(
                series_manifest,
                slice_metadata,
            )
        )

        labels = self._build_label(
            series_manifest,
            slice_metadata,
            dicom_metadata,
        )

        return SeriesMetadata(
            patient_id=slice_metadata[0].patient_id,
            series_id=series_manifest.series_id,
            ordered_slice_keys=ordered_slice_keys,

            dicom=dicom_metadata,
            labels=labels
        )
    