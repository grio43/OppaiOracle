# Minimal warmup + cosine LR scheduler.
# Mirrors the behavior of pl_bolts.optimizers.lr_scheduler.LinearWarmupCosineAnnealingLR
# so you don't need the pl_bolts package.
import math
import warnings
from typing import List
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler


class WarmupStableDecayLR(_LRScheduler):
    """Optimizer-update WSD with a bounded, persisted 1-sqrt cooldown.

    total_steps is a ceiling including cooldown. The fraction is measured
    against elapsed pre-cooldown updates, as specified by the V2 plan.
    """
    def __init__(self, optimizer, *, warmup_steps, total_steps,
                 cooldown_fraction=0.15, min_lr=0.0, last_epoch=-1):
        if not 0.1 <= cooldown_fraction <= 0.2:
            raise ValueError("WSD cooldown fraction must be in [0.10, 0.20]")
        self.warmup_steps = int(warmup_steps)
        self.total_steps = int(total_steps)
        self.cooldown_fraction = float(cooldown_fraction)
        self.budget_cooldown_start = int(self.total_steps / (1 + self.cooldown_fraction))
        if not 0 <= self.warmup_steps < self.budget_cooldown_start:
            raise ValueError("WSD budget must contain warmup, stable training and cooldown")
        self.min_lr = float(min_lr)
        self.base_max_lr = max(group['lr'] for group in optimizer.param_groups)
        if not 0 <= self.min_lr <= min(group['lr'] for group in optimizer.param_groups):
            raise ValueError("WSD lr_end must lie between zero and the stable LR")
        self.cooldown_start = None
        self.cooldown_steps = None
        self.cooldown_reason = None
        self.cooldown_best_metric = float('-inf')
        super().__init__(optimizer, last_epoch)

    @property
    def phase(self):
        if self.cooldown_start is not None:
            return 'complete' if self.finished else 'cooldown'
        return 'warmup' if self.last_epoch < self.warmup_steps else 'stable'

    @property
    def finished(self):
        return (self.cooldown_start is not None
                and self.last_epoch >= self.cooldown_start + self.cooldown_steps)

    def start_cooldown(self, reason='plateau'):
        if self.cooldown_start is not None:
            return False
        if self.last_epoch < self.warmup_steps:
            raise ValueError("Cannot begin cooldown during warmup")
        self.cooldown_start = self.last_epoch
        self.cooldown_steps = min(
            max(1, math.ceil(self.cooldown_fraction * self.last_epoch)),
            self.total_steps - self.last_epoch,
        )
        if self.cooldown_steps < 1:
            raise ValueError("No budget remains for cooldown")
        self.cooldown_reason = reason
        return True

    def get_lr(self):
        if self.cooldown_start is None and self.last_epoch >= self.budget_cooldown_start:
            self.start_cooldown('budget')
        if self.cooldown_start is not None:
            progress = min(1.0, max(0.0, (self.last_epoch - self.cooldown_start) / self.cooldown_steps))
            factor = 1 - math.sqrt(progress)
        elif self.warmup_steps and self.last_epoch < self.warmup_steps:
            # The first update is nonzero; warmup_steps updates reach stable LR.
            factor = (self.last_epoch + 1) / self.warmup_steps
        else:
            factor = 1.0
        return [self.min_lr + (base - self.min_lr) * factor for base in self.base_lrs]

    def load_state_dict(self, state_dict):
        # An accidental reconfiguration must not silently rewrite an active run.
        for key in ('warmup_steps', 'total_steps', 'cooldown_fraction', 'min_lr', 'base_lrs'):
            if state_dict.get(key) != getattr(self, key):
                raise ValueError(f"WSD resume geometry changed: {key}; restore the prepared run configuration")
        super().load_state_dict(state_dict)
        for group, lr in zip(self.optimizer.param_groups, self.get_last_lr()):
            group['lr'] = lr


class LinearWarmupCosineLR(_LRScheduler):
    """
    Linearly warms up from `warmup_start_lr` to each param group's base_lr over `warmup_epochs`,
    then cosine anneals from base_lr to `eta_min` over the remaining epochs up to `max_epochs`.

    Step this scheduler once per epoch.
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
            # scale runs from 0 .. (warmup_epochs-1)/warmup_epochs, so the first epoch
            # uses warmup_start_lr and the cosine phase starts at base_lr at e=warmup_epochs.
            scale = e / max(1, self.warmup_epochs)
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

        # Normal cosine schedule. At e=warmup_epochs (first cosine epoch) t=0 so LR=base_lr;
        # at e=max_epochs-1 t=(total_cosine-1)/total_cosine, approaching eta_min.
        t = (e - self.warmup_epochs) / total_cosine  # 0..~1 over cosine schedule
        t = min(max(t, 0.0), 1.0)
        return [
            self.eta_min + (base_lr - self.eta_min) * (1 + math.cos(math.pi * t)) / 2.0
            for base_lr in base_lrs
        ]

    def _get_closed_form_lr(self):
        # Provide closed-form for compatibility with some PyTorch internals
        return self.get_lr()
