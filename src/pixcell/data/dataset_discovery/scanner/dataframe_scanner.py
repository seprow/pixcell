from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class DataFrameRecord:

    patient_id: str
    series_id: str
    sop_uid: str

    dataframe_index: int

    relative_annotation_path: str | None


class DataFrameScanner:
    """Scan dataframe and build a lightweight index."""

    def __init__(
        self,
        dataframe_path: Path,
        patient_column: str = "dicom_series.PatientID",
        series_column: str = "dicom_series.id",
        sop_column: str = "dicom_series.SOPInstanceUID",
        annotation_column: str = "RelativeAnnotationPath",
    ):
        self.dataframe_path = dataframe_path

        self.patient_column = patient_column
        self.series_column = series_column
        self.sop_column = sop_column
        self.annotation_column = annotation_column

    def scan(self) -> dict[tuple[str, str], DataFrameRecord]:
        df = pd.read_pickle(self.dataframe_path)

        index: dict[tuple[str, str], DataFrameRecord] = {}

        for position, (row_index, row) in enumerate(df.iterrows()): # position(iloc), row_index(loc)

            series_id = str(row[self.series_column])
            sop_uid = str(row[self.sop_column])

            record = DataFrameRecord(
                patient_id=str(row[self.patient_column]),
                series_id=series_id,
                sop_uid=sop_uid,
                dataframe_index=position,
                relative_annotation_path=(
                    None
                    if pd.isna(row[self.annotation_column])
                    else str(row[self.annotation_column])
                ),
            )

            index[(series_id, sop_uid)] = record

        return index