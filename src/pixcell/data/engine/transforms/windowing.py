from __future__ import annotations

import SimpleITK as sitk
from monai.transforms import MapTransform
import numpy as np

class Windowing:

    @staticmethod
    def apply_sitk(
        image: sitk.Image,
        window_center: float,
        window_width: float,
        normalize: bool = True,
    ) -> sitk.Image:

        lower = window_center - (window_width / 2.0)
        upper = window_center + (window_width / 2.0)

        image = sitk.Clamp(
            image,
            lowerBound=lower,
            upperBound=upper,
        )

        if normalize:
            image = sitk.ShiftScale(
                image,
                shift=-lower,
                scale=1.0 / (upper - lower),
            )

        return image
    
    @staticmethod
    def apply_np(
        image: np.ndarray,
        window_center: float,
        window_width: float,
        normalize: bool = False,
    ) -> np.ndarray:

        lower = window_center - (window_width / 2.0)
        upper = window_center + (window_width / 2.0)

        image = np.clip(image, lower, upper)

        if normalize:
            # Min-Max
            image = (image - lower) / (upper - lower)

        return image


class Windowingd(MapTransform):
    """
    Apply CT windowing to image(s).

    Args:
        keys: Keys to apply the transform to.
        window_center: Window center (HU).
        window_width: Window width (HU).
        normalize: If True, scale intensities to [0, 1].
    """

    def __init__(
        self,
        keys,
        window_center: float,
        window_width: float,
        normalize: bool = False,
        allow_missing_keys: bool = False,
    ):
        super().__init__(keys, allow_missing_keys)

        self.window_center = window_center
        self.window_width = window_width
        self.normalize = normalize

        self.lower = window_center - window_width / 2.0
        self.upper = window_center + window_width / 2.0

    def __call__(self, data):
        d = dict(data)

        for key in self.keys:

            image = d[key]

            image = np.clip(image, self.lower, self.upper)

            if self.normalize:
                image = image.astype(np.float32)
                image = (image - self.lower) / (
                    self.upper - self.lower
                )

            d[key] = image

        return d