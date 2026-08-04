from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from pixcell.data.engine.manifests import (
    VolumeImage,
    SliceImage,
)


class BaseImageReader(ABC):

    @abstractmethod
    def read(
        self,
        source: Path,
    ) -> (
        VolumeImage | SliceImage
    ):
        raise NotImplementedError