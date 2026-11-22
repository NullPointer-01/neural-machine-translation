from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler


class NoamScheduler(LRScheduler):
    def __init__(
        self,
        optimizer: Optimizer,
        d_model: int,
        warmup_steps: int = 4000,
        factor: float = 1.0,
        last_epoch: int = -1,
    ) -> None:
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.factor = factor

        # LRScheduler.__init__ calls get_lr() immediately
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step_num = max(self.last_epoch, 1)
        lr = (
            self.factor
            * pow(self.d_model, -0.5)
            * min(
                pow(step_num, -0.5),
                step_num * pow(self.warmup_steps, -1.5),
            )
        )

        return [lr for _ in self.optimizer.param_groups]
