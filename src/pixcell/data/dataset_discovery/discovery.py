from __future__ import annotations

from pathlib import Path

from pixcell.data.dataset_discovery.builders import(
    DatasetBuilder, 
    PatientBuilder, 
    SeriesBuilder, 
    SummaryBuilder
) 
from pixcell.data.dataset_discovery.matcher import SliceMatcher
from pixcell.data.dataset_discovery.manifests import DatasetManifest
from pixcell.data.dataset_discovery.scanner import(
    DataFrameScanner,
    DatasetRootScanner,
    SeriesScanner,
    SliceScanner
    )


class DatasetDiscovery:
    """
    Orchestrates the dataset discovery pipeline.
    """

    def __init__(
        self,
        dataset_root: Path,
        *,
        dicom: bool | None = None,
        annotation: bool | None = None,
        dataframe: bool | None = None,
    ):
        self.dataset_root = dataset_root

        self.dicom = dicom
        self.annotation = annotation
        self.dataframe = dataframe

    def discover(self) -> DatasetManifest:

        # Discover dataset structure
        structure = DatasetRootScanner(self.dataset_root).scan()

        # Discover series
        train_series = SeriesScanner(structure.train_dir).scan()

        annotation_series = {}

        if structure.annotation_dir is not None:
            annotation_series = SeriesScanner(
                structure.annotation_dir
            ).scan()

        # Discover slice files
        dicom_scanner = SliceScanner(".dcm")
        annotation_scanner = SliceScanner(".json")

        dicom_index = {
            series_id: dicom_scanner.scan(series_dir)
            for series_id, series_dir in train_series.items()
        }

        annotation_index = {
            series_id: annotation_scanner.scan(series_dir)
            for series_id, series_dir in annotation_series.items()
        }

        # Discover dataframe
        dataframe_index = {}

        if structure.dataframe_path is not None:
            dataframe_index = DataFrameScanner(
                structure.dataframe_path
            ).scan()

        # Match resources
        slices = SliceMatcher(
            dicom=self.dicom,
            annotation=self.annotation,
            dataframe=self.dataframe,
        ).match(
            dicom_index=dicom_index,
            annotation_index=annotation_index,
            dataframe_index=dataframe_index,
        )

        # Build manifests
        series = SeriesBuilder().build(slices)

        patients = PatientBuilder().build(slices)

        summary = SummaryBuilder().build(
            patients=patients,
            series=series,
            slices=slices,
        )

        # Assemble dataset
        return DatasetBuilder().build(
            name="iaaa-contest-bct",
            structure=structure,
            patients=patients,
            series=series,
            slices=slices,
            summary=summary,
        )