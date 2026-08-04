from __future__ import annotations

from pixcell.data.metadata.extractors import(
    AnnotationMetadataExtractor,
    LabelExtractor,
    DicomMetadataExtractor
    )
from pixcell.data.metadata.manifests import SliceMetadata
from pixcell.data.dataset_discovery.manifests import SliceManifest


class SliceMetadataBuilder:

    def __init__(
        self,
        dicom_extractor: DicomMetadataExtractor,
        annotation_extractor: AnnotationMetadataExtractor,
        label_extractor: LabelExtractor,
    ):
        self._dicom_extractor = dicom_extractor
        self._annotation_extractor = annotation_extractor
        self._dataframe_extractor = label_extractor

    def build(
        self,
        slice_manifest: SliceManifest,
    ) -> SliceMetadata:

        dicom_metadata = self._dicom_extractor.extract(
            slice_manifest.dicom_path
        )

        annotation_metadata = self._annotation_extractor.extract(
            slice_manifest.annotation_path
        )

        label_metadata = self._dataframe_extractor.extract(
            slice_manifest.dataframe_index,
            slice_manifest.annotation_path
        )

        return SliceMetadata(
            patient_id=dicom_metadata.patient_id,
            series_id=slice_manifest.series_id,
            sop_uid=slice_manifest.sop_uid,

            dicom_path=slice_manifest.dicom_path,
            annotation_path=slice_manifest.annotation_path,

            dicom=dicom_metadata,
            annotation=annotation_metadata,
            labels=label_metadata,
        )