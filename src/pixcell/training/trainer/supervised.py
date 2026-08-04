import torch
import time
from tqdm import tqdm
from pathlib import Path

from pixcell.utils import History

from loguru import logger


class SupervisedTrainingLoop:

    def __init__(
        self,
        device,
        config,
        model,
        criterion,
        optimizer,
        metric,
        scheduler=None,
    ):

        self.config = config
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.metric = metric
        self.scheduler = scheduler
        self.device = device
        self.history = History()

        self.model.to(self.device)

        # AMP
        self.use_amp = getattr(config.training, "use_amp", True)
        self.scaler = torch.amp.GradScaler(enabled=self.use_amp)

        self.best_val_loss = float("inf")
        self.no_improve_epochs = 0

        self.checkpoint_dir = Path(config.training.checkpoint_directory)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.start_epoch = 1

    # -------------------------------------------------
    # forward + loss
    # -------------------------------------------------

    def forward_step(self, batch):

        logger.debug("Forward step started")

        images = batch["image"].to(self.device) 

        logger.debug(f"Image shape: {tuple(images.shape)}")

        targets = {
            k: v.to(self.device)
            for k, v in batch["targets"].items()
        }

        with torch.amp.autocast(
            device_type=self.device.type,
            enabled=self.use_amp
        ):

            logger.debug("Model forward...")

            outputs = self.model(images)

            for k, v in outputs.items():
                logger.debug(f"Output {k}: {tuple(v.shape)}")

            logger.debug("Computing loss...")

            total_loss, loss_logs = self.criterion(outputs, targets)

            logger.debug(f"Loss: {total_loss.item():.6f}")

        return outputs, targets, total_loss ,loss_logs

    # -------------------------------------------------
    # train epoch
    # -------------------------------------------------

    def train_epoch(self, loader, epoch):

        logger.info(
            f"Train epoch {epoch} | batches={len(loader)}"
        )

        self.model.train()

        running_logs = {}

        progress = tqdm(
            loader,
            desc=f"Train {epoch}",
            disable=not self.config.training.show_progress
        )

        for batch_idx, batch in enumerate(progress):

            logger.debug(f"Batch {batch_idx} started")

            t0 = time.perf_counter()

            self.optimizer.zero_grad(
                set_to_none=True
            )

            _, _, loss, logs = self.forward_step(batch)

            logger.debug(
                f"Forward finished ({time.perf_counter()-t0:.3f}s)"
            )

            # AMP backward
            self.scaler.scale(loss).backward()

            logger.debug(
                f"Backward finished ({time.perf_counter()-t1:.3f}s)"
            )

            # gradient clipping
            if self.config.training.max_grad_norm > 0:

                self.scaler.unscale_(self.optimizer)

                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.training.max_grad_norm
                )

            t2 = time.perf_counter()

            self.scaler.step(self.optimizer)
            self.scaler.update()

            logger.debug(
                f"Optimizer finished ({time.perf_counter()-t2:.3f}s)"
            )

            if (
                self.scheduler is not None
                and self.config.training.scheduler
                and self.config.training.scheduler.step_per_batch
            ):
                self.scheduler.step()


            for k, v in logs.items():
                running_logs[k] = running_logs.get(k, 0.0) + v.item()

            progress.set_postfix(loss=loss.item())

            if torch.cuda.is_available():

                logger.debug(
                        f"CUDA allocated={torch.cuda.memory_allocated()/1024**3:.2f} GB | "
                        f"reserved={torch.cuda.memory_reserved()/1024**3:.2f} GB"
                    )

            logger.debug(f"Batch {batch_idx} finished")

        train_logs = {
                    f"train/{k}": v / len(loader)
                    for k, v in running_logs.items()
        }

        return train_logs


    # -------------------------------------------------
    # validation
    # -------------------------------------------------

    @torch.no_grad()
    def validate_epoch(self, loader, epoch):

        logger.info(
            f"Validation epoch {epoch} | batches={len(loader)}"
        )

        self.model.eval()
        running_logs = {}

        self.metric.reset()

        progress = tqdm(
            loader,
            desc=f"Val {epoch}",
            disable=not self.config.training.show_progress
        )

        for batch_idx, batch in enumerate(progress):

            logger.debug(f"Validation batch {batch_idx}")

            outputs, targets, loss, logs = self.forward_step(batch)

            for k, v in logs.items():
                running_logs[k] = running_logs.get(k, 0.0) + v.item()

            # update metrics
            self.metric.update(outputs, targets)

            progress.set_postfix(loss=loss.item())

        val_logs = {
            f"val/{k}": v / len(loader)
            for k, v in running_logs.items()
        }

        metrics = self.metric.compute()

        metric_logs = {
            f"val/{k}": v
            for k, v in metrics.items()
        }

        epoch_logs = {}
        epoch_logs.update(val_logs)
        epoch_logs.update(metric_logs)

        return epoch_logs

    # -------------------------------------------------
    # checkpoint
    # -------------------------------------------------


    def save_checkpoint(self, epoch):

        path = self.checkpoint_dir / self.config.training.checkpoint_file

        checkpoint = {
            "epoch": epoch,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scaler": self.scaler.state_dict(),
            "history": self.history.state_dict(),
            "best_val_loss": self.best_val_loss,
            "no_improve_epochs": self.no_improve_epochs,
        }

        if self.scheduler is not None:
            checkpoint["scheduler"] = self.scheduler.state_dict()

        torch.save(checkpoint, path)

        logger.info(f"Saved checkpoint -> {path}")


    def load_checkpoint(self, path):

        checkpoint = torch.load(
            path,
            map_location=self.device,
        )

        self.model.load_state_dict(
            checkpoint["model"]
        )

        self.optimizer.load_state_dict(
            checkpoint["optimizer"]
        )

        self.scaler.load_state_dict(
            checkpoint["scaler"]
        )

        if (
            self.scheduler is not None
            and "scheduler" in checkpoint
        ):
            self.scheduler.load_state_dict(
                checkpoint["scheduler"]
            )

        if "history" in checkpoint:
            self.history.load_state_dict(
                checkpoint["history"]
            )

        self.no_improve_epochs = checkpoint.get(
            "no_improve_epochs",
            0,
        )

        self.start_epoch = checkpoint["epoch"] + 1
        self.best_val_loss = checkpoint["best_val_loss"]

        logger.info(f"Checkpoint loaded <- {path}")

    # -------------------------------------------------
    # training loop
    # -------------------------------------------------

    def fit(self, train_loader, val_loader):

        logger.info("Training started")

        num_epochs = self.config.training.num_epochs
        patience = self.config.training.early_stopping_patience

        for epoch in range(self.start_epoch, num_epochs + 1):

            logger.info(f"========== Epoch {epoch} ==========")

            train_logs = self.train_epoch(train_loader, epoch)

            val_logs = self.validate_epoch(val_loader, epoch)

            logs = {}
            logs.update(train_logs)
            logs.update(val_logs)

            self.history.add_epoch(
                epoch,
                logs,
            )

            logger.info(
                f"Epoch {epoch} | {logs}"
            )

            current = val_logs["val/total"]

            # scheduler per epoch
            if (
                self.scheduler
                and self.config.training.scheduler
                and not self.config.training.scheduler.step_per_batch
            ):
                self.scheduler.step(metric=current)

            if current < self.best_val_loss:

                self.best_val_loss = current
                self.no_improve_epochs = 0

                self.save_checkpoint(epoch)

            else:

                self.no_improve_epochs += 1

                if patience > 0 and self.no_improve_epochs >= patience:

                    logger.info("Early stopping triggered")
                    break

        logger.info("Training finished")
