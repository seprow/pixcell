from __future__ import annotations

from pathlib import Path


class SliceScanner:
    """Discover slice files inside a series directory."""

    def __init__(self, suffix: str):
        self.suffix = suffix.lower()

    def scan(self, series_dir: Path) -> dict[str, Path]:
        if not series_dir.exists():
            return {}

        slices: dict[str, Path] = {}

        for file in series_dir.iterdir():
            if not file.is_file():
                continue

            if file.suffix.lower() != self.suffix:
                continue

            sop_uid = file.stem
            slices[sop_uid] = file

        return dict(sorted(slices.items()))