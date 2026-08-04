from __future__ import annotations

from pathlib import Path
from collections import defaultdict

from  pixcell.data.dataset_discovery.manifests import DatasetManifest
from  pixcell.data.metadata.builders import (
    DatasetMetadataBuilder,
    SeriesMetadataBuilder,
    SliceMetadataBuilder
)
from pixcell.data.metadata.extractors import (
    AnnotationMetadataExtractor,
    LabelExtractor,
    DicomMetadataExtractor,
)
from pixcell.data.metadata.manifests import MetadataManifest
from pixcell.utils import PathResolver
from pixcell.configs import Config


class Metadata:

    def __init__(
        self,
        config: Config
    ):
        self._config = config
        self._path_resolver = PathResolver(self._config)
        self.dataframe_path = self._path_resolver.df_path

        self._slice_builder = SliceMetadataBuilder(
            dicom_extractor=DicomMetadataExtractor(),
            annotation_extractor=AnnotationMetadataExtractor(),
            label_extractor=LabelExtractor(
                self.dataframe_path
            ),
        )

        self._series_builder = SeriesMetadataBuilder(self._config)

        self._dataset_builder = DatasetMetadataBuilder()

        

    def build(
        self,
        dataset_manifest: DatasetManifest,
    ) -> MetadataManifest:

        # Slice Metadata
        slice_metadata = {}

        for key, slice_manifest in dataset_manifest.slices.items():
            slice_metadata[key] = self._slice_builder.build(
                slice_manifest
            )

        # Group by Series
        grouped = defaultdict(list)

        for metadata in slice_metadata.values():
            grouped[metadata.series_id].append(metadata)

        # Series Metadata
        series_metadata = {}

        for series_id, slices in grouped.items():
            series_manifest = dataset_manifest.series[series_id]

            series_metadata[series_id] = (
                self._series_builder.build(
                    series_manifest=series_manifest,
                    slice_metadata=slices,
                )
            )

        # Dataset Metadata
        return self._dataset_builder.build(
            slices=slice_metadata,
            series=series_metadata,
        )