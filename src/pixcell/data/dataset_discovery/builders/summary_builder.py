from __future__ import annotations

from pixcell.data.dataset_discovery.manifests import (
    DatasetSummary,
    PatientManifest,
    SeriesManifest,
    SliceManifest,
)


class SummaryBuilder:
    """
    Build DatasetSummary from discovered manifests.
    """

    def build(
        self,
        patients: dict[str, PatientManifest],
        series: dict[str, SeriesManifest],
        slices: dict[tuple[str, str], SliceManifest],
    ) -> DatasetSummary:

        annotated_series = sum(
            s.num_annotations > 0
            for s in series.values()
        )

        annotated_slices = sum(
            s.has_annotation
            for s in slices.values()
        )

        missing_dicoms = sum(
            not s.has_dicom
            for s in slices.values()
        )

        missing_annotations = sum(
            not s.has_annotation
            for s in slices.values()
        )

        missing_dataframe_rows = sum(
            not s.has_dataframe
            for s in slices.values()
        )

        return DatasetSummary(
            num_patients=len(patients),
            num_series=len(series),
            num_slices=len(slices),

            annotated_series=annotated_series,
            annotated_slices=annotated_slices,

            missing_dicoms=missing_dicoms,
            missing_annotations=missing_annotations,
            missing_dataframe_rows=missing_dataframe_rows,
        )