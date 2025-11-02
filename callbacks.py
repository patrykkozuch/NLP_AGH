import abc

import torch
from torch.optim import Optimizer

from transformer.transformer import Transformer


class Callback(abc.ABC):
    @abc.abstractmethod
    def should_trigger(self, step: int) -> bool:
        ...

    @abc.abstractmethod
    def run(self, step: int):
        ...

class LoggingCallback(Callback):
    def __init__(self, logger, freq: int):
        self.logger = logger
        self.freq = freq

    def should_trigger(self, step: int) -> bool:
        return step % self.freq == 0

    def run(self, step: int):
        self.logger.log(step)

class CheckpointCallback(Callback):
    def __init__(self, transformer: Transformer, optimizer: Optimizer, scheduler, cfg: dict, checkpoint_dir, freq: int):
        self.transformer = transformer
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.cfg = cfg
        self.checkpoint_dir = checkpoint_dir
        self.freq = freq

    def should_trigger(self, step: int) -> bool:
        return step % self.freq == 0

    def run(self, step: int):
        checkpoint_path = self.checkpoint_dir / f"checkpoint_step_{step}.pt"
        torch.save({
                'steps': step,
                'model_state_dict': self.transformer.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict() if hasattr(self.scheduler, 'state_dict') else None,
                'cfg': self.cfg,
            },
            checkpoint_path
        )
