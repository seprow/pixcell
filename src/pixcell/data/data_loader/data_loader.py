from torch.utils.data import DataLoader, Dataset
from pixcell.configs import Config

def build_dataloader(
    dataset: Dataset,
    config: Config,
    training: bool,
) -> DataLoader:
    return DataLoader(
        dataset=dataset,
        batch_size=config.data.batch_size,
        shuffle=training,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory,
        persistent_workers=config.data.num_workers > 0,
        drop_last=training,
    )