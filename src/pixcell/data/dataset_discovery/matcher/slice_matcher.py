from __future__ import annotations

from pathlib import Path

from pixcell.data.dataset_discovery.manifests import SliceManifest
from pixcell.data.dataset_discovery.scanner import DataFrameRecord


class SliceMatcher:
    """Match DICOM files, annotation files and dataframe records."""

    def __init__(
        self,
        *,
        dicom: bool | None = None,
        annotation: bool | None = None,
        dataframe: bool | None = None,
    ):
        self.dicom = dicom
        self.annotation = annotation
        self.dataframe = dataframe

    def _should_include(
        self,
        has_dicom: bool,
        has_annotation: bool,
        has_dataframe: bool,
    ) -> bool:
        return (
            (self.dicom is None or has_dicom == self.dicom)
            and (self.annotation is None or has_annotation == self.annotation)
            and (self.dataframe is None or has_dataframe == self.dataframe)
        )

    def match(
        self,
        dicom_index: dict[str, dict[str, Path]],
        annotation_index: dict[str, dict[str, Path]],
        dataframe_index: dict[tuple[str, str], DataFrameRecord],
    ) -> dict[tuple[str, str], SliceManifest]:

        manifests: dict[tuple[str, str], SliceManifest] = {}

        # Collect every slice existing in any source
        keys = set()

        for series_id, sop_map in dicom_index.items():
            keys.update((series_id, sop_uid) for sop_uid in sop_map)

        for series_id, sop_map in annotation_index.items():
            keys.update((series_id, sop_uid) for sop_uid in sop_map)

        keys.update(dataframe_index.keys())

        # Build SliceManifest
        for series_id, sop_uid in sorted(keys):

            dicom_path = dicom_index.get(series_id, {}).get(sop_uid)

            annotation_path = annotation_index.get(series_id, {}).get(sop_uid)

            df_record = dataframe_index.get((series_id, sop_uid))

            has_dicom=dicom_path is not None

            has_annotation=annotation_path is not None

            has_dataframe=df_record is not None

            if not self._should_include(
                has_dicom,
                has_annotation,
                has_dataframe,
            ):
                continue

            manifests[(series_id, sop_uid)] = SliceManifest(
                patient_id="" if df_record is None else df_record.patient_id,

                series_id=series_id,
                sop_uid=sop_uid,

                dicom_path=dicom_path,
                annotation_path=annotation_path,

                dataframe_index=None
                if df_record is None
                else df_record.dataframe_index,

                has_dicom=has_dicom,
                has_annotation=has_annotation,
                has_dataframe=has_dataframe,
            )

        return manifests