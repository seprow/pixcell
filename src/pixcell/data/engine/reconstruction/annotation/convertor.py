from __future__ import annotations

import numpy as np
import SimpleITK as sitk

from pixcell.configs import Config
from pixcell.data.engine.reconstruction.image import (
    DicomSliceReader,
    DicomVolumeReader,
)
from pixcell.utils import PathResolver
from pixcell.data.engine.manifests import SliceId


class NumpyToSitkConverter:

    def __init__(
        self,
        config: Config | None = None,
    ) -> None:
        self._config = config 

        self._path_resolver = PathResolver(self._config)

        self._slice_reader = DicomSliceReader()
        self._volume_reader = DicomVolumeReader()

    def convert(
        self,
        annotation: np.ndarray,
        slice_ids: SliceId | list[SliceId],
    ) -> sitk.Image:

        if isinstance(slice_ids, tuple):

            series_id, slice_id = slice_ids

            reference = self._slice_reader.read(
                self._path_resolver.slice_path(
                    series_id,
                    slice_id,
                )
            ).image

        else:
            paths = [
                self._path_resolver.slice_path(
                    series_id,
                    slice_id,
                )
                for series_id, slice_id in slice_ids
            ]

            reference = self._volume_reader.read(
                paths
            ).image

        image = sitk.GetImageFromArray(
            annotation.astype(np.uint8)
        )

        image.CopyInformation(reference)

        return image