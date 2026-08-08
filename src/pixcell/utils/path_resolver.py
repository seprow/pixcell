from pathlib import Path
from pixcell.configs import Config

class PathResolver:

    def __init__(self, config: Config):

        self.dicom_dir: Path = config.data_dir / "training"
        self.annotation_dir: Path = config.data_dir / "annotations"
        self.df_path: Path = config.data_dir / "training_df.pkl"

        self.processed_root = (
            config.processed_data_dir
            / config.data.dataset_subset
        )
        self.dataset_dir = self.processed_root / "dataset"
        self.metadata_dir = self.processed_root / "metadata"
        self.image_dir = self.processed_root / "image"
        self.annotation_output_dir = (
            self.processed_root / "annotation" 
        )

    # ---------- dicom ----------

    def series_path(self, series_id: str) -> Path:  
        return self.dicom_dir / series_id

    def slice_path(self, series_id: str, slice_id: str) -> Path:
        return self.series_path(series_id) / f"{slice_id}.dcm"

    # ---------- annotations ----------

    def annotation_series_path(self, annotation_series_id: str) -> Path:
        return self.annotation_dir / annotation_series_id

    def annotation_slice_path(
        self,
        annotation_series_id: str,
        annotation_slice_id: str
    ) -> Path:
        return self.annotation_series_path(annotation_series_id) / f"{annotation_slice_id}.json"

    # ---------- processed_data ----------

    # Dataset / Metadata
    def dataset_path(self) -> Path:
        return self.dataset_dir / "dataset.pkl"

    def metadata_path(self) -> Path:
        return self.metadata_dir / "metadata.pkl"

    # Image cache
    def image_volume_path(
        self,
        series_id: str,
    ) -> Path:
        return (
            self.image_dir
            / "volume"
            / f"{series_id}.npy"
        )

    def image_slice_path(
        self,
        series_id: str,
        slice_id: str,
    ) -> Path:
        return (
            self.image_dir
            / "slice"
            / series_id
            / f"{slice_id}.npy"
        )

    # Annotation cache
    def annotation_volume_path(
        self,
        task: str,
        builder: str,
        series_id: str,
        key: None
    ) -> Path:
        
        filename = (
            f"{series_id}.npy"
            if key is None
            else f"{series_id}-{key}.npy"
        )
        return (
            self.annotation_output_dir
            / task
            / builder
            / filename
        )

    def annotation_slice_path_cache(
        self,
        task: str,
        builder: str,
        series_id: str,
        slice_id: str,
        key: None
    ) -> Path:
        
        filename = (
            f"{slice_id}.npy"
            if key is None
            else f"{slice_id}-{key}.npy"
        )
        return (
            self.annotation_output_dir
            / task
            / builder
            / series_id
            / filename
        )
    
