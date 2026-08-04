from __future__ import annotations
import SimpleITK as sitk

from pixcell.configs.data_config import ResampleConfig
from pixcell.utils.registry import RESAMPLE_PIPELINE

_INTERPOLATORS = {
    "nearest": sitk.sitkNearestNeighbor,
    "linear": sitk.sitkLinear,
    "bspline": sitk.sitkBSpline,
}

@RESAMPLE_PIPELINE.register("image_spacing_resampler")
class ImageSpacingResampler:
    """
    Resample an image to the target voxel spacing.
    """

    def apply(
        self,
        image: sitk.Image,
        config: ResampleConfig,
    ) -> sitk.Image:

        if config.spacing is None:
            return image

        input_spacing = image.GetSpacing()

        import numpy as np

        if np.allclose(
            input_spacing,
            config.spacing,
            atol=1e-3,
        ):
            return image

        input_size = image.GetSize()

        output_size = [
            int(
                round(
                    input_size[i]
                    * input_spacing[i]
                    / config.spacing[i]
                )
            )
            for i in range(3)
        ]

        resampler = sitk.ResampleImageFilter()

        resampler.SetInterpolator(
            _INTERPOLATORS[
                config.interpolator.lower()
            ]
        )

        resampler.SetOutputSpacing(
            config.spacing
        )

        resampler.SetSize(output_size)

        resampler.SetOutputOrigin(
            image.GetOrigin()
        )

        resampler.SetOutputDirection(
            image.GetDirection()
        )

        resampler.SetTransform(
            sitk.Transform()
        )

        resampler.SetDefaultPixelValue(config.default_pixel_value)

        return resampler.Execute(image)