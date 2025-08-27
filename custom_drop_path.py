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
        if keep_prob == 0.0:
            return x.new_zeros(x.shape)
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor
