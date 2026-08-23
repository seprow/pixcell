from __future__ import annotations

import torch

from pixcell.configs import Config
from pixcell.data.engine.reconstruction.builder import Reconstruction
from pixcell.data import (
    patient_level_split,
    patient_level_cv,
)
from pixcell.data.data_loader.dataset import (
    DataHandler,
    SupervisedDataset,
)
from pixcell.data.data_loader import build_dataloader
from pixcell.training.loss import build_loss
from pixcell.training.optimizer import OptimizerBuilder
from pixcell.evaluation.metrics import MetricBuilder, postprocessor
from pixcell.training.scheduler import *
from pixcell.training.trainer import SupervisedTrainingLoop
from pixcell.utils import PathResolver
from pixcell.utils.registry import (
    TRANSFORM_PIPELINE,
    MODEL_REGISTRY,
    LR_SCHEDULERS
)

from loguru import logger
from pixcell.utils import setup_logger

class SupervisedRunner:

    def __init__(self, config: Config):

        self.config = config

        setup_logger(
            log_level=config.logging.log_level,
            log_file=config.log_dir / config.logging.log_file,
            log_to_file=config.logging.log_to_file,
            log_to_console=config.logging.log_to_console,
        )

        logger.info("Initializing runner...")

        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )

        self.path = PathResolver(config)

        self.dataset, self.metadata = Reconstruction(config).build()

        self.train_transform = self._build_transform("train")
        self.val_transform = self._build_transform("val")


    def _build_transform(self, mode: str):

        train_transform_pipeline = self.config.data.train_transform
        val_transform_pipeline = self.config.data.val_transform

        if mode == "train":

            if train_transform_pipeline is None:
                return None

            return TRANSFORM_PIPELINE.build(train_transform_pipeline)

        else:

            if val_transform_pipeline is None:
                return None

            return TRANSFORM_PIPELINE.build(val_transform_pipeline)


    def _build_trainer(self):

        model = MODEL_REGISTRY.build(
            self.config.model.model
        )

        criterion = build_loss(
            self.config
        )

        optimizer = OptimizerBuilder.build(
            model,
            self.config,
        )


        if self.config.training.scheduler is not None:
            scheduler = LR_SCHEDULERS.build(
                self.config.training.scheduler.name,
                optimizer=optimizer,
                **self.config.training.scheduler.params,
            )


        metric = MetricBuilder(
            self.config,
            self.device,
        )

        trainer = SupervisedTrainingLoop(
            device=self.device,
            config=self.config,
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            metric=metric,
            scheduler=scheduler,
        )

        if self.config.training.resume:
            trainer.load_checkpoint(
                self.config.training.checkpoint_directory / self.config.training.checkpoint_file
            )

        return trainer


    def _build_splits(self):

        patient_ids = list(
            self.dataset.patients.keys()
        )

        if self.config.data.train_val_split is not None:

            return [
                patient_level_split(
                    patient_ids,
                    self.config,
                )
            ]

        return patient_level_cv(
            patient_ids,
            self.config,
        )


    def run(self):

        for fold, (train_ids, val_ids) in enumerate(
            self._build_splits(),
            start=1,
        ):

            print(f'\n========== Fold {fold} ==========')

            train_handler = DataHandler(
                train_ids,
                self.dataset,
                self.metadata,
                self.path,
            )

            val_handler = DataHandler(
                val_ids,
                self.dataset,
                self.metadata,
                self.path,
            )

            train_dataset = SupervisedDataset(
                train_handler,
                self.config,
                self.train_transform,
            )

            val_dataset = SupervisedDataset(
                val_handler,
                self.config,
                self.val_transform,
            )

            train_loader = build_dataloader(
                train_dataset,
                self.config,
                training=True,
            )

            val_loader = build_dataloader(
                val_dataset,
                self.config,
                training=False,
            )

            trainer = self._build_trainer()

            trainer.fit(
                train_loader,
                val_loader,
            )