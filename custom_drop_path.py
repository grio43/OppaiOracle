import torch
import torch.nn as nn

class SafeDropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        drop_prob = float(drop_prob)
        if not (0.0 <= drop_prob < 1.0):
            raise ValueError(f"drop_prob must be in [0, 1), got {drop_prob}")
        self.drop_prob = drop_prob

    def forward(self, x):
        if not self.training or self.drop_prob == 0.0:
            return x
        keep_prob = 1.0 - self.drop_prob
        # keep_prob is always > 0 due to __init__ validation
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        # Draw in float32: drawing in x.dtype (e.g. bf16) quantizes keep_prob
        # and skews the realized keep rate. The 0/1 mask casts back exactly.
        random_tensor = keep_prob + torch.rand(shape, dtype=torch.float32, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor.to(x.dtype)
