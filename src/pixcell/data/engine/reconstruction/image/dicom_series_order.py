from __future__ import annotations
from typing import Union, Optional
from pathlib import Path
import SimpleITK as sitk

from loguru import logger


class DicomSeriesOrder:
    """
    Determine the correct DICOM slice order using GDCM.
    """

    def get_ordered_paths(
        self,
        series_directory: Union[Path, str],
        series_instance_uid: Optional[str] = None
    ) -> list[Path] | None:
        """
        Returns the ordered DICOM file paths for a series.

        Parameters
        ----------
        series_directory
            Directory containing DICOM slices.

        Returns
        -------
        list[Path]
            Ordered DICOM file paths.
        """

        if series_instance_uid is None:
            series_ids = sitk.ImageSeriesReader.GetGDCMSeriesIDs(str(series_directory))
            if not series_ids:
                logger.warning(
                    f"No DICOM series found in directory {series_directory}"
                    )
                return None
            series_id = series_ids[0]
        
        else:
            series_id = series_instance_uid

        series_file_names = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(
            str(series_directory), 
            series_id,
        )

        if not series_file_names:
            logger.warning(
                f"No DICOM files found for series ID {series_id} in directory {series_directory}"
                )
            return None

        return [Path(p) for p in series_file_names]

        