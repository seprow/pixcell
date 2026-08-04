from __future__ import annotations

import numpy as np
import SimpleITK as sitk
import pydicom

from pixcell.data.metadata.manifests import (
    SliceMetadata,
    SeriesMetadata,
)


class HUConverter:
    """
    Convert a CT volume to Hounsfield Units (HU).

    sitk.ImageSeriesReader, automatically handles HU conversion. 
    """

    def convert(
        self,
        image: sitk.Image,
        metadata: SliceMetadata | SeriesMetadata,
        slope: float | None = None,
        intercept: float | None = None,
    ) -> sitk.Image:

        if slope is None:
            if isinstance(metadata, SliceMetadata):
                slope = metadata.dicom.rescale_slope
            else:
                slope = metadata.dicom.rescale_slope # TODO: Add RescaleIntercept to DicomMetadata

        if intercept is None:
            if isinstance(metadata, SliceMetadata):
                intercept = metadata.dicom.rescale_intercept
            else:
                intercept = metadata.dicom.rescale_intercept # TODO: Add RescaleIntercept to DicomMetadata

        slope = 1.0 if slope is None else slope
        intercept = 0.0 if intercept is None else intercept

        image = sitk.Cast(
            image,
            sitk.sitkFloat32,
        )

        return image * slope + intercept
    

def dicom_to_hu(ds: pydicom.Dataset) -> np.ndarray:
    """Convert DICOM pixel array to Hounsfield Units."""
    
    img = ds.pixel_array.astype(np.float32)

    intercept = getattr(ds, "RescaleIntercept", 0.0)
    slope = getattr(ds, "RescaleSlope", 1.0)

    hu = img * slope + intercept
    return hu