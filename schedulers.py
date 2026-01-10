# Minimal, PL 2.x-compatible warmup + cosine LR scheduler.
# Mirrors the behavior of pl_bolts.optimizers.lr_scheduler.LinearWarmupCosineAnnealingLR
# so you don't need the pl_bolts package.
import math
import warnings
from typing import List
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler


class LinearWarmupCosineLR(_LRScheduler):
    """
    Linearly warms up from `warmup_start_lr` to each param group's base_lr over `warmup_epochs`,
    then cosine anneals from base_lr to `eta_min` over the remaining epochs up to `max_epochs`.

    Step this scheduler once per epoch. In PyTorch Lightning, set
    `interval="epoch"` when returning it from `configure_optimizers`.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_epochs: int,
        max_epochs: int,
        warmup_start_lr: float = 0.0,
        eta_min: float = 0.0,
        last_epoch: int = -1,
    ):
        if max_epochs <= 0:
            raise ValueError("max_epochs must be > 0")
        if warmup_epochs < 0:
            raise ValueError("warmup_epochs must be >= 0")
        if warmup_epochs > max_epochs:
            raise ValueError("warmup_epochs cannot exceed max_epochs")

        # Warn about edge case
        if warmup_epochs == max_epochs:
            warnings.warn(
                f"warmup_epochs ({warmup_epochs}) equals max_epochs ({max_epochs}). "
                f"This means the entire training will be warmup with no cosine annealing. "
                f"Learning rate will reach base_lr but never anneal to eta_min={eta_min}. "
                f"This is likely a configuration error. "
                f"Consider setting warmup_epochs < max_epochs to enable cosine schedule.",
                UserWarning,
                stacklevel=2
            )

        self._warmup_epochs = int(warmup_epochs)
        self.max_epochs = int(max_epochs)
        self.warmup_start_lr = float(warmup_start_lr)
        self.eta_min = float(eta_min)
        super().__init__(optimizer, last_epoch)

    # Maintain backward-compat attribute name used in logic
    @property
    def warmup_epochs(self) -> int:
        return self._warmup_epochs

    def get_lr(self) -> List[float]:
        e = self.last_epoch  # epoch index starting at 0 after first step()
        base_lrs = self.base_lrs

        # Warmup phase: linear increase from warmup_start_lr -> base_lr
        if e < self.warmup_epochs:
            # e runs from 0..warmup_epochs-1
            # scale runs from 1/warmup_epochs .. warmup_epochs/warmup_epochs
            scale = (e + 1) / max(1, self.warmup_epochs)
            return [
                self.warmup_start_lr + (base_lr - self.warmup_start_lr) * scale
                for base_lr in base_lrs
            ]

        # Cosine phase: from base_lr -> eta_min across remaining epochs
        total_cosine = self.max_epochs - self.warmup_epochs

        # Edge case: No cosine phase (warmup == max_epochs)
        if total_cosine <= 0:
            # No cosine phase - return eta_min (training at minimum LR after warmup)
            return [self.eta_min for _ in base_lrs]

        # Normal cosine schedule
        t = (e - self.warmup_epochs + 1) / total_cosine  # 0..1 over cosine schedule
        t = min(max(t, 0.0), 1.0)
        return [
            self.eta_min + (base_lr - self.eta_min) * (1 + math.cos(math.pi * t)) / 2.0
            for base_lr in base_lrs
        ]

    def _get_closed_form_lr(self):
        # Provide closed-form for compatibility with some PyTorch internals
        return self.get_lr()

