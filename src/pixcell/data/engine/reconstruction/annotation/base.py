from __future__ import annotations
from abc import ABC, abstractmethod
from pathlib import Path

from pixcell.data.engine.manifests import (
    SliceAnnotation,
    VolumeAnnotation,
)


class BaseAnnotationBuilder(ABC):

    @abstractmethod
    def build(
        self,
        source: Path | list[Path],
    ) -> (
        SliceAnnotation
        | VolumeAnnotation
    ):
        raise NotImplementedError