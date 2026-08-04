from torch.optim.lr_scheduler import LRScheduler


class WarmupDecayScheduler(LRScheduler):

    def __init__(self, optimizer, warmup_steps, total_steps, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        super().__init__(optimizer, last_epoch)

    def get_lr(self):

        step = self.last_epoch + 1

        if step < self.warmup_steps:
            scale = step / self.warmup_steps
        else:
            scale = max(
                0.0,
                (self.total_steps - step) /
                (self.total_steps - self.warmup_steps)
            )

        return [base_lr * scale for base_lr in self.base_lrs]
