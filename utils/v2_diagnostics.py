"""Reporting-only V2 diagnostics; none of these scalars selects checkpoints."""
import torch


def binned_diagnostics(confmat, class_ap, keep, tag_names, frequencies):
    # Reuse the AP estimator's threshold grid: no second N x labels buffer.
    tp = confmat[:, :, 1, 1].float()
    denominator = 2 * tp + confmat[:, :, 0, 1] + confmat[:, :, 1, 0]
    oracle = (2 * tp / denominator.clamp_min(1)).amax(dim=0)
    results = {'val/oracle_f1_macro_binned': oracle[keep].mean().item() if keep.any() else 0.0}
    # Stable equal-tag-count deciles, sorted by distinct-image frequency then name.
    content = sorted((i for i, name in enumerate(tag_names) if not name.startswith('rating:')),
                     key=lambda i: (frequencies.get(tag_names[i], 0), tag_names[i]))
    for decile in range(10):
        indices = content[len(content) * decile // 10:len(content) * (decile + 1) // 10]
        indices = torch.tensor(indices, device=class_ap.device, dtype=torch.long)
        indices = indices[keep[indices]]
        results[f'val_decile/{decile + 1}/mAP'] = class_ap[indices].mean().item() if len(indices) else 0.0
        results[f'val_decile/{decile + 1}/supported_tags'] = len(indices)
    return results
