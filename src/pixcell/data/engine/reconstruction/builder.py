from __future__ import annotations

from pathlib import Path

from pixcell.configs import Config
from pixcell.data.dataset_discovery import DatasetDiscovery
from pixcell.data.metadata import Metadata
from pixcell.data.engine.manifests import (
    SliceImage,
    VolumeImage,
    SliceAnnotation,
    VolumeAnnotation,
    SliceAnnotations,
    VolumeAnnotations,
)
from pixcell.utils import (
    PathResolver,
    PickleIO,
    NumpyIO,
)
from pixcell.data.engine.reconstruction.annotation import (
    segmentation,
    heatmap,
    NumpyToSitkConverter
)
from pixcell.data.engine.reconstruction import resampling
from pixcell.utils.registry import (
    IMAGE_READER_REGISTRY,
    ANNOTATION_BUILDER_REGISTRY,
    RESAMPLE_PIPELINE,
)


_SUBSET_FILTERS = {
    "all_slices": {},

    "annotated_and_labeled_slices": dict(
        dicom=True,
        annotation=True,
        dataframe=True,
    ),

    "labeled_slices": dict(
        dicom=True,
        dataframe=True,
    ),

    "unannotated_labeled_slices": dict(
        dicom=True,
        annotation=False,
        dataframe=True,
    ),
}


class Reconstruction:

    def __init__(
        self,
        config: Config,
    ):
        self._config = config
        self._paths = PathResolver(config)
        self._converter = NumpyToSitkConverter(self._config)


    def build(self):

        dataset = self._load_dataset()

        metadata = self._load_metadata(dataset)

        if not self._config.data.reconstruction.enabled:
            return dataset, metadata

        image_reader = IMAGE_READER_REGISTRY.build(
            self._config.data.reconstruction.image.reader
        )

        annotation_builders = [
            ANNOTATION_BUILDER_REGISTRY.build(
                target.builder,
                **target.parameters,
            )
            for target in self._config.data.reconstruction.annotation.targets
        ]

        image_resampler = None
        annotation_resampler = None

        if self._config.data.reconstruction.resample.enabled:

            image_resampler = RESAMPLE_PIPELINE.build(
                self._config.data.reconstruction.resample.pipeline
            )

            annotation_resampler = RESAMPLE_PIPELINE.build(
                self._config.data.reconstruction.resample.pipeline
            )

        is_volume = (
            "volume"
            in self._config.data.reconstruction.image.reader
        )

        iterable = (
            metadata.series.values()
            if is_volume
            else metadata.slices.values()
        )

        images = {}
        annotations = {}

        for item in iterable:

            image = self._build_image(
                image_reader,
                item,
            )

            if image_resampler is not None:

                image = image.map(
                    lambda img, _: image_resampler.apply(
                        img,
                        self._config.data.reconstruction.resample.image,
                    )
                )


            if self._config.data.reconstruction.cache:
                self._save_image(image)

            if self._config.data.reconstruction.load:
                images[self._key(image)] = image

            for builder, target in zip(
                annotation_builders,
                self._config.data.reconstruction.annotation.targets,
            ):

                annotation = self._build_annotation(
                    builder,
                    target,
                    item,
                )

                if annotation is None:
                    continue

                if annotation_resampler is not None:

                    annotation = annotation.map(
                        lambda ann, _: annotation_resampler.apply(
                            ann,
                            self._config.data.reconstruction.resample.annotation,
                        )
                    )


                if self._config.data.reconstruction.cache:
                    self._save_annotation(
                        annotation,
                        target.task,
                        target.builder,
                        target.cache_key,
                    )

                if self._config.data.reconstruction.load:
                    annotations.setdefault(
                        self._key(annotation),
                        {},
                    )[target.task] = annotation

        if self._config.data.reconstruction.load:
            return dataset, metadata, images, annotations
        
        else: 
            return dataset, metadata
        
    def _load_dataset(self):

        path = self._paths.dataset_path()

        if path.exists():
            return PickleIO.load(path)

        filters = _SUBSET_FILTERS[
            self._config.data.dataset_subset
        ]

        dataset = DatasetDiscovery(
            self._config.data.data_directory,
            **filters,
        ).discover()

        PickleIO.save(
            dataset,
            path,
        )

        return dataset

    def _load_metadata(
        self,
        dataset,
    ):

        path = self._paths.metadata_path()

        if path.exists():
            return PickleIO.load(path)

        metadata = Metadata(
            self._config,
        ).build(dataset)

        PickleIO.save(
            metadata,
            path,
        )

        return metadata

    def _build_image(
        self,
        reader,
        item,
    ):

        if hasattr(item, "ordered_slice_keys"):

            paths = []

            for series_id, slice_id in item.ordered_slice_keys:

                paths.append(
                    self._paths.slice_path(
                        series_id=series_id,
                        slice_id=slice_id,
                    )
                )

            return reader.read(
                paths
            )

        return reader.read(item.dicom_path)

    def _build_annotation(
        self,
        builder,
        target,
        item,
    ):

        if hasattr(item, "ordered_slice_keys"):

            paths = []

            for series_id, slice_id in item.ordered_slice_keys:

                path = self._paths.annotation_slice_path(
                    series_id,
                    slice_id,
                )
                
                if path.exists():
                    paths.append(path)

            if not paths:
                return None
            
            annotation = builder.build(
                paths,
            )

        else:

            if item.annotation_path is None:
                return None

            annotation = builder.build(
                item.annotation_path,
            )

        annotation = annotation.map(self._converter.convert)

        return annotation

    def _save_image(
        self,
        image,
    ):

        if isinstance(
            image,
            VolumeImage,
        ):

            path = self._paths.image_volume_path(
                image.ordered_slice_ids[0][0]
            )

        else:

            path = self._paths.image_slice_path(
                *image.slice_id
            )

        NumpyIO.save(
            image.image,
            path,
        )

    def _save_annotation(
        self,
        annotation,
        task,
        builder,
        cache_key
    ):
        
        annotation.save(self._paths, task, builder, cache_key)


    @staticmethod
    def _key(item):

        if hasattr(
            item,
            "ordered_slice_ids",
        ):
            return item.ordered_slice_ids[0][0]

        return item.slice_id