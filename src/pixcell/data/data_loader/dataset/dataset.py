from __future__ import annotations

from torch.utils.data import Dataset

from pixcell.data.data_loader.dataset import DataHandler
from pixcell.configs import Config
from .target_filters import _TARGET_FILTER
from pixcell.configs.data_config import AnnotationConfig
from pixcell.utils import NumpyIO


class SupervisedDataset(Dataset):

    def __init__(
        self,
        handler: DataHandler,
        config: Config,
        transform=None,
    ):
        super().__init__()

        self.handler = handler
        self._config = config
        self.targets = self._config.training.targets
        self.series_wise = (
            "volume"
            in self._config.data.reconstruction.image.reader
        )
        self.transform = transform

    def __len__(self):
        if self.series_wise:
            return len(self.handler.series_ids)

        return len(self.handler.slice_ids)

    def __getitem__(self, index):

        if self.series_wise:

            image = NumpyIO.load(
                self.handler.series_image_path(index)
            )

            label = self.handler.series_label(index)

        else:

            image = NumpyIO.load(
                self.handler.slice_image_path(index)
            )

            label = self.handler.slice_label(index)


        outputs = {}

        for target in self.targets:

            target_name = target.target_name
            target_info = _TARGET_FILTER[target_name]

            # ---------------- Annotation ----------------

            if isinstance(target_info, AnnotationConfig):

                if self.series_wise:

                    path = self.handler.series_annotation_path(
                        index=index,
                        task=target_info.task,
                        builder=target_info.builder,
                        key=target_info.cache_key,
                    )

                else:

                    path = self.handler.slice_annotation_path(
                        index=index,
                        task=target_info.task,
                        builder=target_info.builder,
                        key=target_info.cache_key,
                    )

                outputs[target_name] = NumpyIO.load(path)

            # ---------------- Label ----------------

            else:

                outputs[target_name] = getattr(
                    label,
                    target_info,
                )

        sample = {
            "image": image,
            **outputs,
        }

        if self.transform is not None:
            sample = self.transform(sample)

        image = sample.pop("image")

        return {
            "image": image, 
            "targets": sample
        }