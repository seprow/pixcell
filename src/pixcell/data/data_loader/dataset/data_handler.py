from pathlib import Path

from pixcell.data.dataset_discovery.manifests import DatasetManifest
from pixcell.data.metadata.manifests import (
    DicomMetadata,
    SeriesLevelDicomMetadata,
    MetadataManifest, 
    LabelMetadata
)
from pixcell.utils import PathResolver

class DataHandler:

    def __init__(
            self,
            patient_ids: list[str],
            dataset: DatasetManifest,
            metadata: MetadataManifest,   
            path_resolver: PathResolver 
    ):
        self.patient_ids = patient_ids
        self._dataset = dataset
        self._metadata = metadata
        self._path_resolver = path_resolver

        self.series_ids = self._patients_to_series_ids()
        self.slice_ids = self._series_to_slice_ids()

    # Bulk API

    @property
    def series_wise_image_path(self) -> list[Path]:
        paths = []

        for sid in self.series_ids:
            paths.append(
                self._path_resolver.image_volume_path(sid)
            )
        return paths
    
    @property
    def slice_wise_image_path(self) -> list[Path]:
        paths = []

        for sid, sop_uid in self.slice_ids:
            paths.append(
                self._path_resolver.image_slice_path(sid, sop_uid)
            )
        return paths           
    
    def series_wise_annotation_path(
            self, 
            task, 
            builder, 
            key=None
    ) -> list[Path]:
        
        paths = []
    
        for sid in self.series_ids:
            paths.append(
                self._path_resolver.annotation_volume_path(
                    series_id=sid,
                    task=task,
                    builder=builder,
                    key=key
                )
            )
        return paths
    
    def slice_wise_annotation_path(
            self, 
            task, 
            builder, 
            key=None
    )-> list[Path]:
        
        paths = []
    
        for sid, sop_uid in self.slice_ids:
            paths.append(
                self._path_resolver.annotation_slice_path_cache(
                    series_id=sid,
                    slice_id=sop_uid,
                    task=task,
                    builder=builder,
                    key=key
                )
            )
        return paths      

    @property
    def series_wise_labels(self) -> list[LabelMetadata]:
        
        return [
            self._metadata.series[series_id].labels
            for series_id in self.series_ids
        ]
    
    @property
    def slice_wise_labels(self) -> list[LabelMetadata]:

        return [
            self._metadata.slices[slice_id].labels
            for slice_id in self.slice_ids
        ]
    
    # Index API (for torch Dataset)

    def series_image_path(self, index: int) -> Path:
        return self._path_resolver.image_volume_path(
            self.series_ids[index]
        )

    def slice_image_path(self, index: int) -> Path:
        series_id, slice_id = self.slice_ids[index]
        return self._path_resolver.image_slice_path(
            series_id,
            slice_id,
        )

    def series_annotation_path(
        self,
        index: int,
        task: str,
        builder: str,
        key: str | None = None,
    ) -> Path:
        return self._path_resolver.annotation_volume_path(
            series_id=self.series_ids[index],
            task=task,
            builder=builder,
            key=key,
        )

    def slice_annotation_path(
        self,
        index: int,
        task: str,
        builder: str,
        key: str | None = None,
    ) -> Path:
        series_id, slice_id = self.slice_ids[index]
        return self._path_resolver.annotation_slice_path_cache(
            series_id=series_id,
            slice_id=slice_id,
            task=task,
            builder=builder,
            key=key,
        )

    def series_label(self, index: int) -> LabelMetadata:
        return self._metadata.series[
            self.series_ids[index]
        ].labels

    def slice_label(self, index: int) -> LabelMetadata:
        return self._metadata.slices[
            self.slice_ids[index]
        ].labels
    
    def metadata_slice(self, index: int) -> DicomMetadata:
        return self._metadata.slices[
            self.slice_ids[index]
        ].dicom

    def metadata_series(self, index: int) -> SeriesLevelDicomMetadata:
        return self._metadata.series[
            self.series_ids[index]
        ].dicom

    # helper

    def _patients_to_series_ids(self) -> list[str]:

        series_ids = []

        for patient_id in self.patient_ids:
            series_ids.extend(self._dataset.patients[patient_id].series_ids)

        return series_ids

    def _series_to_slice_ids(self) -> list[tuple[str, str]]:
        
        slice_ids = []

        for series_id in self.series_ids:
            slice_ids.extend(
                self._metadata.series[series_id].ordered_slice_keys
            )

        return slice_ids