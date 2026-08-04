from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import SimpleITK as sitk

from pixcell.utils import NumpyIO


SliceId = tuple[str, str]

#========================================================
# Image Reconstruction
#========================================================

@dataclass(slots=True)
class VolumeImage:

    image: sitk.Image

    ordered_slice_ids: list[SliceId]

    def map(self, fn):
        return VolumeImage(
            image=fn(
                self.image,
                self.ordered_slice_ids,
            ),
            ordered_slice_ids=self.ordered_slice_ids,
        )

@dataclass(slots=True)
class SliceImage:

    image: sitk.Image

    slice_id: SliceId

    def map(self, fn):
        return SliceImage(
            image=fn(
                self.image,
                self.slice_id,
            ),
            slice_id=self.slice_id,
        )

#========================================================
# Annotation Reconstruction
#========================================================

# Single annotation

@dataclass(slots=True)
class SliceAnnotation:

    annotation: np.ndarray

    slice_id: SliceId

    def map(self, fn):
        return SliceAnnotation(
            annotation=fn(
                self.annotation,
                self.slice_id,
            ),
            slice_id=self.slice_id,
        )
    
    def save(self, path_resolver, task, builder, cache_key=None):


        NumpyIO.save(
            self.annotation,
            path_resolver.annotation_slice_path_cache(
                series_id=self.slice_id[0],
                slice_id=self.slice_id[1],
                task=task,
                builder=builder,
                key=cache_key
            )
        )

@dataclass(slots=True)
class VolumeAnnotation:

    annotation: np.ndarray

    ordered_slice_ids: list[SliceId]

    def map(self, fn):
        return VolumeAnnotation(
            annotation=fn(
                self.annotation,
                self.ordered_slice_ids,
            ),
            ordered_slice_ids=self.ordered_slice_ids,
        )
    
    def save(self, path_resolver, task, builder, cache_key=None):

        NumpyIO.save(
            self.annotation,
            path_resolver.annotation_volume_path(
                series_id=self.ordered_slice_ids[0][0],
                task=task,
                builder=builder,
                key=cache_key
            )
        )

# Multiple annotations (per class)

@dataclass(slots=True)
class SliceAnnotations:

    annotation: dict[int, np.ndarray]

    slice_id: SliceId

    def map(self, fn):
        return SliceAnnotations(
            annotation={
                cls: fn(mask, self.slice_id)
                for cls, mask in self.annotation.items()
            },
            slice_id=self.slice_id,
        )
    
    def save(self, path_resolver, task, builder, cache_key=None):
        for key, value in self.annotation.items():
            NumpyIO.save(
                value,
                path_resolver.annotation_slice_path_cache(
                    series_id=self.slice_id[0],
                    slice_id=self.slice_id[1],
                    task=task,
                    builder=builder,
                    key=key,
                )
            )

@dataclass(slots=True)
class VolumeAnnotations:

    annotation: dict[int, np.ndarray]

    ordered_slice_ids: list[SliceId]

    def map(self, fn):
        return VolumeAnnotations(
            annotation={
                cls: fn(
                    volume,
                    self.ordered_slice_ids,
                )
                for cls, volume in self.annotation.items()
            },
            ordered_slice_ids=self.ordered_slice_ids,
        )
    
    def save(self, path_resolver, task, builder, cache_key=None):
        for key, value in self.annotation.items():
            NumpyIO.save(
                value,
                path_resolver.annotation_volume_path(
                    series_id=self.ordered_slice_ids[0][0],
                    task=task,
                    builder=builder,
                    key=key,
                )
            )