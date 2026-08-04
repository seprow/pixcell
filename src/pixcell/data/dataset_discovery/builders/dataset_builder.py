from __future__ import annotations

from pathlib import Path

from pixcell.data.dataset_discovery.manifests import (
    DatasetManifest,
    DatasetSummary,
    PatientManifest,
    SeriesManifest,
    SliceManifest,
    DatasetStructure
)


class DatasetBuilder:
    """
    Assemble the final DatasetManifest.
    """

    def build(
        self,
        name: str,
        structure: DatasetStructure,
        patients: dict[str, PatientManifest],
        series: dict[str, SeriesManifest],
        slices: dict[tuple[str, str], SliceManifest],
        summary: DatasetSummary,
    ) -> DatasetManifest:

        return DatasetManifest(
            name=name,
            structure=structure,
            patients=patients,
            series=series,
            slices=slices,
            summary=summary,
        )