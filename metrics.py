#!/usr/bin/env python3
"""
Evaluation metrics for multi‑label classification.

This module provides simple functions to compute precision, recall, F1 and
mean average precision for multi‑label predictions.  The pad class at index 0
is excluded from all computations by default.
"""

from __future__ import annotations
from typing import Dict, Iterable, Tuple
import torch


def compute_precision_recall_f1(
    probabilities: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
    pad_index: int = 0,
) -> Dict[str, float]:
    """
    Compute micro‑averaged precision, recall and F1 score for multi‑label
    predictions at a given threshold.

    Args:
        probabilities: Tensor of shape (B, C) with predicted probabilities.
        targets: Tensor of shape (B, C) with binary ground‑truth labels.
        threshold: Threshold above which a probability is considered a positive prediction.
        pad_index: Index of the pad class to ignore.  Both probabilities and
            targets will have this column removed before computation.

    Returns:
        Dictionary with keys ``precision``, ``recall`` and ``f1``.
    """
    if probabilities.dim() != 2 or targets.dim() != 2:
        raise ValueError("probabilities and targets must be 2D tensors")
    # Remove pad column
    probs = probabilities.clone()
    targs = targets.clone()

    # Validate bounds before concatenation
    if pad_index is not None and 0 <= pad_index < probs.size(1):
        if pad_index == 0:
            probs = probs[:, 1:]
            targs = targs[:, 1:]
        elif pad_index == probs.size(1) - 1:
            probs = probs[:, :pad_index]
            targs = targs[:, :pad_index]
        else:
            probs = torch.cat([probs[:, :pad_index], probs[:, pad_index+1:]], dim=1)
            targs = torch.cat([targs[:, :pad_index], targs[:, pad_index+1:]], dim=1)
    preds = (probs >= threshold).float()
    true = targs.float()
    tp = (preds * true).sum()
    fp = ((preds == 1) & (true == 0)).float().sum()
    fn = ((preds == 0) & (true == 1)).float().sum()
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    return {'precision': precision.item(), 'recall': recall.item(), 'f1': f1.item()}


def average_precision_score(
    probabilities: torch.Tensor,
    targets: torch.Tensor,
    pad_index: int = 0,
) -> Dict[str, float]:
    """
    Compute mean average precision (mAP) across classes excluding the pad class.

    A simple approximation of average precision is computed by sorting samples
    by predicted probability for each class and summing the precision
    increments at each true positive.  Classes with no positive examples are
    skipped.

    Args:
        probabilities: Tensor of shape (B, C) with predicted probabilities.
        targets: Tensor of shape (B, C) with binary ground‑truth labels.
        pad_index: Index of the pad class to ignore.

    Returns:
        Dictionary with keys ``map`` (mean AP across classes) and
        ``per_class_ap`` (list of APs for each non‑pad class).
    """
    if probabilities.dim() != 2 or targets.dim() != 2:
        raise ValueError("probabilities and targets must be 2D tensors")
    probs = probabilities.clone()
    targs = targets.clone()

    # Validate bounds before concatenation
    if pad_index is not None and 0 <= pad_index < probs.size(1):
        if pad_index == 0:
            probs = probs[:, 1:]
            targs = targs[:, 1:]
        elif pad_index == probs.size(1) - 1:
            probs = probs[:, :pad_index]
            targs = targs[:, :pad_index]
        else:
            probs = torch.cat([probs[:, :pad_index], probs[:, pad_index+1:]], dim=1)
            targs = torch.cat([targs[:, :pad_index], targs[:, pad_index+1:]], dim=1)
    B, C = probs.shape
    ap_values = []
    for c in range(C):
        scores = probs[:, c]
        labels = targs[:, c]
        # Skip classes with no positive examples
        total_pos = labels.sum().item()
        if total_pos == 0:
            continue
        # Sort by predicted score descending
        sorted_indices = torch.argsort(scores, descending=True)
        sorted_labels = labels[sorted_indices]
        cum_tp = sorted_labels.cumsum(dim=0).float()
        precisions = cum_tp / torch.arange(1, len(cum_tp) + 1, device=probs.device).float()
        recalls = cum_tp / total_pos
        # Integrate precision over recall using step function
        ap = 0.0
        prev_recall = 0.0
        for p, r in zip(precisions, recalls):
            dr = r.item() - prev_recall
            if dr > 0:
                ap += p.item() * dr
                prev_recall = r.item()
        ap_values.append(ap)
    mean_ap = sum(ap_values) / len(ap_values) if ap_values else 0.0
    return {'map': mean_ap, 'per_class_ap': ap_values}