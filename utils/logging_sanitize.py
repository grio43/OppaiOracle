# =========================
# utils/logging_sanitize.py
# =========================
from typing import Any, Dict, Optional
import math

def _to_safe_float(x: Any, reduce: str = "mean") -> Optional[float]:
    """
    Convert a metric (tensor/array/number) to a finite Python float on CPU.
    - If x is a tensor with more than 0 dims, reduce it ('mean' or 'sum') before item().
    - Returns None if result is NaN/Inf or if x is empty.
    """
    try:
        import torch  # local import so this file works without torch if needed
        is_torch = True
    except Exception:
        is_torch = False
        torch = None  # type: ignore

    # ---- tensor path
    if is_torch and torch.is_tensor(x):
        # Detach from graph, move to CPU if needed
        x = x.detach()
        if x.numel() == 0:
            return None
        if x.is_cuda:
            x = x.cpu()
        # Reduce to scalar if necessary
        if x.ndim > 0:
            if reduce == "mean":
                x = x.float().mean()
            elif reduce == "sum":
                x = x.float().sum()
            else:
                # Fallback: take mean
                x = x.float().mean()
        # Extract Python float
        x = x.item()

    # ---- non-tensor path
    else:
        try:
            x = float(x)
        except Exception:
            return None

    # ---- final finite check
    return x if math.isfinite(x) else None


def sanitize_metrics(metrics: Dict[str, Any], reduce: str = "mean") -> Dict[str, float]:
    """
    Convert a dict of metrics into CPU Python floats and drop non-finite values.
    """
    clean: Dict[str, float] = {}
    for k, v in metrics.items():
        fv = _to_safe_float(v, reduce=reduce)
        if fv is not None:
            clean[k] = fv
    return clean

def ensure_finite_tensor(x):
    """
    Replace non-finite tensors with zeros (same device/dtype) to avoid backprop explosions,
    while preserving the computation graph.
    """
    import torch
    if torch.isfinite(x).all():
        return x
    # Return a tensor of zeros with the same graph as the input
    return x * 0

def assert_all_finite(metrics: Dict[str, Any], where: str = ""):
    import math
    import torch
    bad = []
    for k, v in metrics.items():
        if torch.is_tensor(v):
            v = v.detach()
            if v.numel() == 0:
                bad.append(k)
                continue
            if not torch.isfinite(v).all():
                bad.append(k)
        else:
            try:
                f = float(v)
                if not math.isfinite(f):
                    bad.append(k)
            except Exception:
                bad.append(k)
    if bad:
        raise ValueError(f"Non-finite metrics {bad} {('at ' + where) if where else ''}")


def safe_div(numer, denom, eps: float = 1e-12):
    """
    Guard against divide-by-zero in custom loss normalizations.
    """
    import torch
    if torch.is_tensor(denom):
        denom = denom.clamp_min(eps)
    else:
        denom = denom if denom != 0 else eps
    return numer / denom
