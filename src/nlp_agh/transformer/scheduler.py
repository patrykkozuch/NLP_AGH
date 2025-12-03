from typing import Optional

from torch.optim.lr_scheduler import LRScheduler


class TransformerLRScheduler(LRScheduler):
    """
    Learning rate scheduler from 'Attention Is All You Need' paper.

    lrate = d_model^(-0.5) * min(step_num^(-0.5), step_num * warmup_steps^(-1.5))

    Args:
        optimizer: PyTorch optimizer
        d_model: Model dimension
        warmup_steps: Number of warmup steps (default: 4000)
    """

    def __init__(self, optimizer, d_model, warmup_steps=4000):
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.step_num = 1
        self.optimizer = optimizer
        super().__init__(optimizer)

    def step(self, epoch: Optional[int] = None):
        """Update learning rate after each training step."""
        self.step_num += 1
        lr = self._get_lr()
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def _get_lr(self):
        """Calculate learning rate based on current step."""
        step = self.step_num
        lr = (self.d_model ** -0.5) * min(
            step ** -0.5,
            step * (self.warmup_steps ** -1.5)
        )
        return lr

    def get_last_lr(self):
        """Return the last computed learning rate."""
        return [self._get_lr()]