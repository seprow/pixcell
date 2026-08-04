from __future__ import annotations

from pixcell.data.metadata.manifests import MetadataManifest, SeriesMetadata, SliceMetadata


class DatasetMetadataBuilder:

    def build(
        self,
        slices: dict[tuple[str, str], SliceMetadata],
        series: dict[str, SeriesMetadata],
    ) -> MetadataManifest:

        return MetadataManifest(
            series=series,
            slices=slices,
        )