from __future__ import annotations
from pathlib import Path
import SimpleITK as sitk

from pixcell.utils.registry import IMAGE_READER_REGISTRY
from pixcell.data.engine.manifests import SliceImage
from pixcell.data.engine.reconstruction.image import (
    BaseImageReader,
)

@IMAGE_READER_REGISTRY.register("dicom_slice_reader")
class DicomSliceReader(BaseImageReader):

    def read(
        self,
        source: Path,
    ) -> SliceImage:

        if not isinstance(source, Path):
            raise TypeError(
                "DicomSliceReader expects a file Path."
            )

        image = sitk.ReadImage(str(source))

        if image.GetDimension() == 3 and image.GetSize()[2] == 1:
            image = image[:, :, 0]

        return SliceImage(
            image=image,
            slice_id=(source.parent.name, source.stem),
        )