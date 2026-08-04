from __future__ import annotations
from pathlib import Path


class DicomPathMatcher:
    """
    Match ordered DICOM paths against a reference path list while
    preserving the original order.
    """

    def match(
        self,
        ordered_paths: list[Path],
        reference_paths: list[Path],
    ) -> list[Path]:
   
        reference_stems = {
            path.stem
            for path in reference_paths
        }

        return [
            path
            for path in ordered_paths
            if path.stem in reference_stems
        ]