from __future__ import annotations

from collections import defaultdict

from pixcell.data.dataset_discovery.manifests import SeriesManifest, SliceManifest


class SeriesBuilder:
    """
    Build SeriesManifest objects from SliceManifest objects.
    """

    def build(
        self,
        slices: dict[tuple[str, str], SliceManifest],
    ) -> dict[str, SeriesManifest]:

        grouped: dict[str, list[SliceManifest]] = defaultdict(list)

        for slice_manifest in slices.values():
            grouped[slice_manifest.series_id].append(slice_manifest)

        series_manifests: dict[str, SeriesManifest] = {}

        for series_id, slice_list in grouped.items():

            dicom_directory = next(
                (
                    s.dicom_path.parent
                    for s in slice_list
                    if s.dicom_path is not None
                ),
                None,
            )

            annotation_directory = next(
                (
                    s.annotation_path.parent
                    for s in slice_list
                    if s.annotation_path is not None
                ),
                None,
            )

            slice_keys = [
                (s.series_id, s.sop_uid)
                for s in sorted(slice_list, key=lambda x: x.sop_uid)
            ]

            num_dicoms = sum(s.has_dicom for s in slice_list)

            num_annotations = sum(s.has_annotation for s in slice_list)

            series_manifests[series_id] = SeriesManifest(
                series_id=series_id,
                dicom_directory=dicom_directory,
                annotation_directory=annotation_directory,
                slice_keys=slice_keys,
                num_dicoms=num_dicoms,
                num_annotations=num_annotations,
            )

        return series_manifests