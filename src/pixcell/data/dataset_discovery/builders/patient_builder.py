from __future__ import annotations

from collections import defaultdict

from pixcell.data.dataset_discovery.manifests import PatientManifest, SliceManifest


class PatientBuilder:
    """
    Build PatientManifest objects from SliceManifest objects.
    """

    def build(
        self,
        slices: dict[tuple[str, str], SliceManifest],
    ) -> dict[str, PatientManifest]:

        patient_series: dict[str, set[str]] = defaultdict(set)

        for slice_manifest in slices.values():

            if not slice_manifest.patient_id:
                continue

            patient_series[slice_manifest.patient_id].add(
                slice_manifest.series_id
            )

        patient_manifests: dict[str, PatientManifest] = {}

        for patient_id, series_ids in patient_series.items():

            patient_manifests[patient_id] = PatientManifest(
                patient_id=patient_id,
                series_ids=sorted(series_ids),
            )

        return patient_manifests