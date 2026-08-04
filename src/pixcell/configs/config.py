"""
Configuration management.

Uses dataclasses for type-safe configuration.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from pathlib import Path
import os
from dacite import (
    from_dict,
    Config as DaciteConfig,
)

import SimpleITK as sitk

from .data_config import *
from .training_config import *


@dataclass
class DataConfig:
    """Configuration for data loading and preprocessing.

    This class controls how the iaaa-contest-bct dataset is loaded, preprocessed,
    and split into training/validation sets.

    Attributes:
        data_directory: 
            Directory where the raw dataset is downloaded and stored. 
                If ``Config.relative_path`` is ``True``:
                this path is interpreted relative to ``Config.project_root``.
                Otherwise: it is treated as an absolute or user-provided path.
        dataset_subset:
            - all_slices -> Unsupervised/ Semi-supervised
            - labeled_slices -> Supervised (label)
            - annotated_and_labeled_slices -> Supervised (annotation + labels + post-processing)
            - unannotated_labeled_slices -> Post-processing test
        train_val_split: 
            Fraction of training data to use for training (rest goes to validation).
        cv_n_splits:
            number of cross validation folds. 
            Exactly one of `cv_n_splits` or `train_val_split` must be specified.
        seed: 
            Random seed for reproducibility.
        use_augmentation: 
            Whether to apply data augmentation during training.
        batch_size: 
            Number of samples per batch during training/validation/testing.
        num_workers: 
            Number of worker processes for data loading.
            More workers can speed up data loading but use more memory.
            Set to 0 to disable multiprocessing (useful for debugging).

    """

    data_directory: Path | str = Path(r"S:\work\Projects\PixCell\data")
    processed_data_directory: Path | str = Path(r"S:\work\Projects\PixCell\processed_data")

    dataset_subset: str = "all_slices" 

    reconstruction: ReconstructionConfig = field(
        default_factory=ReconstructionConfig
    )

    train_val_split: Optional[float] = None
    cv_n_splits: Optional[int] = None
    seed: int = 42

    train_transform: Optional[str] = None
    val_transform: Optional[str] = None
   
    batch_size: int = 64
    num_workers: Optional[int] = None
    pin_memory: bool = True
    

@dataclass
class ModelConfig:
    """
    Configuration for model architecture.

    """

    model: str = "unet"


@dataclass
class TrainingConfig:
    """Configuration for model training process.

    This class controls the training loop, optimization, and checkpointing.

    Attributes:
        checkpoint_directory: Directory where model checkpoints are saved.
            If ``Config.relative_path`` is ``True``, this path is interpreted
            relative to ``Config.project_root``.
        checkpoint_file: 
            Name of the checkpoint file stored inside ``checkpoint_directory``.
        learning_mode:
            - supervised
            - semi_supervised
            - unsupervised
        show_progress: 
            Whether to show progress bars during training/validation.
        use_amp:
            Enable Automatic Mixed Precision (AMP) during training.
        max_grad_norm: 
            Maximum gradient norm for gradient clipping.
            Prevents exploding gradients by clipping gradients that exceed this value.
            Set to 0.0 to disable gradient clipping.
        num_epochs: 
            Number of complete passes through the training dataset.
        early_stopping_patience: 
            Number of epochs to wait before stopping if validation
            accuracy doesn't improve.
            Set to 0 to disable early stopping.
            Example: patience=5 means stop if no improvement for 5 consecutive epochs.
    """

    checkpoint_directory: str = "checkpoints"
    checkpoint_file: str = "best_model.pt"
    resume: bool = False

    learning_mode: str = "supervised"

    show_progress: bool = True

    use_amp: bool = True

    max_grad_norm: float = 1.0

    targets: list[TargetConfig] = field(default_factory=list)

    loss: LossConfig = field(default_factory=LossConfig)

    optimizer: OptimizerConfig = field(
        default_factory=OptimizerConfig
    )

    scheduler: Optional[SchedulerConfig] = None

    metrics: list[TargetMetricConfig] = field(default_factory=list)

    num_epochs: int = 15
    
    early_stopping_patience: int = 5
    


@dataclass
class LoggingConfig:
    """Configuration for logging and output.

    This class controls how training progress and information is logged.

    Attributes:
        log_directory: 
            Directory where log files are stored. 
            If ``Config.relative_path`` is ``True``, this path is interpreted
            relative to ``Config.project_root``.
        log_file: 
            Name of the log file stored inside ``log_directory``. 
        log_level:                              
            Logging verbosity level.
            Options: 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'.
            DEBUG = most verbose (all messages), INFO = standard (default),
            WARNING = warnings and errors only, ERROR = errors only.
        log_to_file: 
            Whether to write logs to a file.
            If True, logs are saved to log_file. Useful for keeping training history.
        log_to_console: 
            Whether to print logs to console/terminal.
            If True, logs are displayed in real-time during training.
            Useful for monitoring progress without checking log files.
    """
    
    log_directory: str = "logs"
    log_file: str = "training.log"
    log_level: str = "INFO"
    log_to_file: bool = True
    log_to_console: bool = True


@dataclass
class Config:
    """Main configuration class combining all sub-configs.

    This is the top-level configuration that combines DataConfig, ModelConfig,
    TrainingConfig, and LoggingConfig. It also manages project paths.

    Attributes:
        relative_path: Whether configured directories are interpreted relative
            to ``project_root``. If ``False``, the configured paths are used
            directly.
        project_root: Root directory of the project. Used as the base directory
            when ``relative_path`` is ``True``.
        data: Data loading and preprocessing configuration.
            Controls how the dataset is loaded, split, and preprocessed.
        model: Model architecture configuration.
            Defines the neural network structure (layer sizes, dropout, etc.).
        training: Training process configuration.
            Controls optimization, epochs, learning rate, checkpointing, etc.
        logging: Logging configuration.
            Controls log output, verbosity, and file/console logging.
        project_root: Root directory of the project.
            Automatically detected based on config file location.
            Used as base for all other paths.
        data_dir: Directory where iaaa-contest-bct dataset is stored.
            Automatically set to: project_root / data.root
            Created automatically if it doesn't exist.
        checkpoint_dir: Directory where model checkpoints are saved.
            Automatically set to: project_root / training.checkpoint_dir
            Created automatically if it doesn't exist.
        log_dir: Directory where log files are saved.
            Automatically set to: project_root / 'logs'
            Created automatically if it doesn't exist.
    """
    
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    # Paths
    relative_path: bool = False
    project_root: Path = field(
        default_factory=lambda: Path(__file__).parent.parent.parent.parent
    )
    data_dir: Path = field(init=False)
    processed_data_dir: Path = field(init=False)
    checkpoint_dir: Path = field(init=False)
    log_dir: Path = field(init=False)

    def __post_init__(self):
        """Initialize derived paths after object creation.

        This method is automatically called after the dataclass is instantiated.
        It computes the full paths for data_dir, checkpoint_dir, and log_dir
        based on project_root and the respective config values.
        It also creates these directories if they don't exist.
        """

        if self.relative_path:
            # Compute full paths relative to project root
            self.data_dir = self.project_root / self.data.data_directory
            self.processed_data_dir = self.project_root / self.data.processed_data_directory
            self.checkpoint_dir = self.project_root / self.training.checkpoint_directory
            self.log_dir = self.project_root / self.logging.log_directory

        else:
            self.data_dir = Path(self.data.data_directory)
            self.processed_data_dir = Path(self.data.processed_data_directory)
            self.checkpoint_dir = Path(self.training.checkpoint_directory)
            self.log_dir = Path(self.logging.log_directory) 
            

        # Create directories if they don't exist
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_dict(cls, config_dict: dict) -> "Config":

        return from_dict(
            data_class=cls,
            data=config_dict,
            config=DaciteConfig(
                strict=True,
                check_types=True,
                type_hooks={
                    Path: Path,
                },
                cast=[tuple, Path],
            ),
        )



def get_default_config() -> Config:
    """Get default configuration.

    Returns:
        Config instance with all default values.
    """
    return Config()















