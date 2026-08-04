from __future__ import annotations
from pathlib import Path
import SimpleITK as sitk

from pixcell.utils.registry import IMAGE_READER_REGISTRY
from pixcell.data.engine.manifests import VolumeImage
from pixcell.data.engine.reconstruction.image import (
    BaseImageReader,
)

@IMAGE_READER_REGISTRY.register("dicom_volume_reader")
class DicomVolumeReader(BaseImageReader):

    def read(
        self,
        source: list[Path]
    ) -> VolumeImage:


        reader = sitk.ImageSeriesReader()

        reader.SetFileNames(
            [str(path) for path in source]
        )

        #reader.MetaDataDictionaryArrayUpdateOn()
        #reader.LoadPrivateTagsOn()

        image = reader.Execute()

        ordered_slice_ids = [
            (path.parent.name, path.stem)
            for path in source
        ]

        return VolumeImage(
            image=image,
            ordered_slice_ids=ordered_slice_ids,
        )

