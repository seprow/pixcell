from __future__ import annotations
from pathlib import Path
from pydicom import dcmread
from pydicom.tag import Tag

from pixcell.data.metadata.manifests import DicomMetadata


class DicomMetadataExtractor:
    """
    Extract DICOM header metadata without loading pixel data.
    """

    def extract(self, dicom_path: Path) -> DicomMetadata:

        ds = dcmread(dicom_path, stop_before_pixels=True)

        return DicomMetadata(
            
            patient_id=self._get(ds, (0x0010,0x0020)),
            frame_of_reference_uid=self._get(ds, (0x0020,0x0052)),

            rows=self._to_int(self._get(ds, (0x0028,0x0010))),
            columns=self._to_int(self._get(ds, (0x0028,0x0011))),

            pixel_spacing=self._get_pixel_spacing(ds),

            slice_thickness=self._to_float(
                self._get(ds, (0x0018,0x0050))
            ),

            image_orientation_patient=self._get_orientation(ds),

            image_position_patient=self._get_position(ds),

            slice_location=self._to_float(
                self._get(ds, (0x0020,0x1041))
            ),

            instance_number=self._to_int(
                self._get(ds, (0x0020,0x0013))
            ),

            # Computed later by SeriesMetadataBuilder
            slice_spacing=None,

            rescale_intercept=self._to_float(
                self._get(ds, (0x0028,0x1052))
            ),

            rescale_slope=self._to_float(
                self._get(ds, (0x0028,0x1053))
            ),

            window_center=self._get_first_float(
                self._get(ds, (0x0028,0x1050))
            ),

            window_width=self._get_first_float(
                self._get(ds, (0x0028,0x1051))
            ),

            bits_allocated=self._to_int(
                self._get(ds, (0x0028,0x0100))
            ),

            bits_stored=self._to_int(
                self._get(ds, (0x0028,0x0101))
            ),

            pixel_representation=self._to_int(
                self._get(ds, (0x0028,0x0103))
            ),

            samples_per_pixel=self._to_int(
                self._get(ds, (0x0028,0x0002))
            ),

            photometric_interpretation=self._get(
                ds,
                (0x0028,0x0004),
            ),

            image_type=self._get_image_type(ds),

            manufacturer=self._get(ds, (0x0008,0x0070)),

            manufacturer_model_name=self._get(
                ds,
                (0x0008,0x1090),
            ),

            software_versions=self._get(
                ds,
                (0x0018,0x1020),
            ),

            modality=self._get(ds, (0x0008,0x0060)),

            kvp=self._to_float(self._get(ds, (0x0018,0x0060))),

            exposure_time=self._to_float(
                self._get(ds, (0x0018,0x1150))
            ),

            xray_tube_current=self._to_float(
                self._get(ds, (0x0018,0x1151))
            ),

            convolution_kernel=self._get(
                ds,
                (0x0018,0x1210),
            ),

            protocol_name=self._get(
                ds,
                (0x0018,0x1030),
            ),

            patient_position=self._get(
                ds,
                (0x0018,0x5100),
            ),

            reconstruction_diameter=self._to_float(
                self._get(ds, (0x0018,0x1100))
            ),

            gantry_detector_tilt=self._to_float(
                self._get(ds, (0x0018,0x1120))
            ),

            spiral_pitch_factor=self._to_float(
                self._get(ds, (0x0018,0x9311))
            ),

            ctdi_vol=self._to_float(
                self._get(ds, (0x0018,0x9345))
            ),

            patient_birth_date=self._get(
                ds,
                (0x0010,0x0030),
            ),

            patient_sex=self._get(ds, (0x0010,0x0040)),
        )


    @staticmethod
    def _get(ds, tag: Tag):

        element = ds.get(tag)
        return None if element is None else element.value

    @staticmethod
    def _to_float(value):

        if value is None:
            return None

        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _to_int(value):

        if value is None:
            return None

        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    def _get_first_float(self, value):

        if value is None:
            return None

        if isinstance(value, (list, tuple)):
            value = value[0]

        return self._to_float(value)

    def _get_pixel_spacing(self, ds):

        value = self._get(ds, (0x0028,0x0030))

        if value is None:
            return None

        return (
            float(value[0]),
            float(value[1]),
        )

    def _get_orientation(self, ds):

        value = self._get(ds, (0x0020,0x0037))

        if value is None:
            return None

        return tuple(float(v) for v in value)

    def _get_position(self, ds):

        value = self._get(ds, (0x0020,0x0032))

        if value is None:
            return None

        return (
            float(value[0]),
            float(value[1]),
            float(value[2]),
        )

    def _get_image_type(self, ds):

        value = self._get(ds, (0x0008,0x0008))

        if value is None:
            return None

        return tuple(value)