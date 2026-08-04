from __future__ import annotations
from pathlib import Path


class SeriesScanner:
    """Discover all series folders inside a dataset directory."""

    def __init__(self, series_root: Path):
        self.series_root = series_root

    def scan(self) -> dict[str, Path]:
        if not self.series_root.exists():
            raise FileNotFoundError(f"{self.series_root} does not exist.")

        if not self.series_root.is_dir():
            raise NotADirectoryError(f"{self.series_root} is not a directory.")

        series = {}

        for path in self.series_root.iterdir():
            if path.is_dir():
                series[path.name] = path

        return dict(sorted(series.items()))