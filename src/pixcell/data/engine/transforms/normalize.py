from __future__ import annotations

import numpy as np
import SimpleITK as sitk

from pixcell.config import NormalizeConfig


class IntensityNormalizer:
    """
    Normalize image intensities.

    Supported methods
    -----------------
    none
    minmax
    zscore
    percentile
    """

    def apply(
        self,
        image: sitk.Image,
        config: NormalizeConfig,
    ) -> sitk.Image:

        if (
            not config.enabled
            or config.method == "none"
        ):
            return image

        array = sitk.GetArrayFromImage(image).astype(
            np.float32
        )

        method = config.method.lower()

        if method == "minmax":

            minimum = array.min()
            maximum = array.max()

            if maximum > minimum:
                array = (
                    array - minimum
                ) / (maximum - minimum)

        elif method == "zscore":

            mean = array.mean()
            std = array.std()

            if std > 0:
                array = (
                    array - mean
                ) / std

        elif method == "percentile":

            p1 = np.percentile(array, 1)
            p99 = np.percentile(array, 99)

            array = np.clip(
                array,
                p1,
                p99,
            )

            if p99 > p1:
                array = (
                    array - p1
                ) / (p99 - p1)

        else:

            raise ValueError(
                f"Unsupported normalization method: {config.method}"
            )

        output = sitk.GetImageFromArray(array)

        output.CopyInformation(image)

        return output