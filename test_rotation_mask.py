"""
test_rotation_mask.py

Verify apply_random_rotation rotates the padding mask together with the canvas.

Before the 2026-07-29 fix the mask was left static while the canvas rotated. That
is wrong in both directions, and the larger term was the one the old rationale
did not cover: rotation tilts the whole letterbox, so real content rotates up
into the region a static mask still calls a bar, and pixel_to_token_ignore
(threshold 0.9) then drops it from attention.

READ THIS BEFORE ADDING CHECKS HERE. apply_random_rotation re-asserts pad_color
under the mask it produces (`rotated.paste(pad_color, mask=mask_img)`), so
"pixels agree with the mask" is MANUFACTURED and cannot validate the mask
geometry -- a mask rotated by the wrong sign, wrong centre or wrong angle still
satisfies it. Geometry must be checked against a ground truth built without the
implementation's mask path. [3] does that with a flat sentinel-colour canvas
rotated as RGB, and asserts the result is sign-discriminating.

Tests:
  1. Contract: shape/dtype/semantics preserved, mask changes, input not mutated.
  2. Invariant: masked-pad pixels are exactly pad_color (guaranteed by the
     paste; recorded because process_image_cpu promises it, NOT as geometry).
  3. Geometry vs. an INDEPENDENT ground truth: token-exact, sign-discriminating,
     and the old static-mask behaviour measurably fails it.
  4. Determinism, and the RNG stream advances by exactly 2 draws.
  5. Angle sampling: over many seeds the produced mask matches the truth for the
     drawn angle, |angle| stays in [min,max], and the sign is balanced.
  6. End-to-end through SidecarJsonDataset with rotation ON.

Run from repo root with the project's Python:
    L:/Dab/payton_env/Scripts/python.exe test_rotation_mask.py
"""

import json
import random
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from dataset_loader import apply_random_rotation, process_image_cpu, SidecarJsonDataset
from mask_utils import ensure_pixel_padding_mask, pixel_to_token_ignore
from vocabulary import TagVocabulary

TARGET = 448
PATCH = 16
THRESH = 0.9
PAD = (114, 114, 114)
# Sentinel palette for the independent ground truth: flat colours survive the
# LANCZOS content downscale unchanged, and neither equals the other.
C_CONTENT = (7, 200, 13)
C_PAD = (250, 3, 251)
FAILURES = []


def check(cond, label, detail=""):
    if cond:
        print(f"  PASS  {label}")
    else:
        print(f"  FAIL  {label} {detail}")
        FAILURES.append(label)


def make_portrait(w=365, h=512):
    """Median-aspect booru-like image (short/long = 0.713, the measured corpus
    median) with structure at the content edges so dropped content is visible."""
    rng = np.random.default_rng(0)
    a = rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)
    a[:8, :, :] = 255
    a[-8:, :, :] = 0
    return Image.fromarray(a, mode="RGB")


def make_flat(color, w=365, h=512):
    return Image.fromarray(
        np.full((h, w, 3), np.asarray(color, dtype=np.uint8), dtype=np.uint8), "RGB")


def token_ignore(pmask):
    m = ensure_pixel_padding_mask(pmask.unsqueeze(0), mask_semantics="pad")
    return pixel_to_token_ignore(m, PATCH, THRESH).squeeze(0)


def independent_truth(angle, w=365, h=512):
    """Ground-truth (canvas, pad_mask) for `angle`, built WITHOUT touching the
    implementation's mask code: letterbox a flat sentinel colour, rotate the RGB
    canvas, and read pad back off the pixels."""
    canvas, pmask = process_image_cpu(make_flat(C_CONTENT, w, h), TARGET, C_PAD)
    rot = canvas.rotate(angle, resample=Image.NEAREST, expand=False, fillcolor=C_PAD)
    truth = torch.from_numpy(np.all(np.array(rot) == np.asarray(C_PAD), axis=-1))
    return canvas, pmask, truth


def expected_angle(seed, lo, hi):
    """Mirror the implementation's documented 2-draw sampling to know which angle
    (and sign) a given seed produces. The 2-draw contract is itself asserted
    independently in [4]."""
    random.seed(seed)
    a = random.uniform(lo, hi)
    return -a if random.random() < 0.5 else a


# ---------------------------------------------------------------- 1
print("\n[1/6] contract: shape, dtype, semantics, mask changes, no mutation")
img = make_portrait()
canvas0, pmask0 = process_image_cpu(img, TARGET, PAD)
snapshot = pmask0.clone()
random.seed(11)
rot, rpm = apply_random_rotation(canvas0, pmask0, 2.0, 5.0, PAD)

check(isinstance(rot, Image.Image) and rot.size == (TARGET, TARGET),
      "canvas shape preserved", f"got {getattr(rot, 'size', None)}")
check(rpm.shape == pmask0.shape, "mask shape preserved", f"got {tuple(rpm.shape)}")
check(rpm.dtype == torch.bool, "mask dtype bool", f"got {rpm.dtype}")
check(rpm.is_contiguous(), "mask contiguous")
check(not torch.equal(rpm, snapshot), "mask actually rotated (differs from static)")
check(torch.equal(pmask0, snapshot), "input mask NOT mutated in place")
# True=PAD. Semantic inversion would read ~0.713 for this fixture, so this band
# catches a flipped convention.
check(0.10 < rpm.float().mean().item() < 0.45,
      "mask pad fraction sane (True=PAD, not inverted)",
      f"frac={rpm.float().mean().item():.3f}")

# ---------------------------------------------------------------- 2
print("\n[2/6] pad_color invariant (manufactured by the paste -- not geometry)")
arr = np.array(rot)
pad_px = arr[rpm.numpy()]
check(pad_px.size > 0 and np.all(pad_px == np.asarray(PAD)),
      "every masked-pad pixel is exactly pad_color (no bicubic ringing)",
      f"unique={np.unique(pad_px.reshape(-1, 3), axis=0)[:3].tolist()}")
content_px = arr[~rpm.numpy()]
frac_padcolor = np.all(content_px == np.asarray(PAD), axis=-1).mean()
check(frac_padcolor < 0.10, "content region is not wholesale pad_color",
      f"frac_padcolor_in_content={frac_padcolor:.3f}")

# ---------------------------------------------------------------- 3
print("\n[3/6] geometry vs INDEPENDENT truth (sign-discriminating)")
for mag in (2.0, 3.5, 5.0, 8.0):
    SEED = 0
    ang = expected_angle(SEED, mag, mag)          # min==max -> |angle| == mag
    canvas_s, pmask_s, truth = independent_truth(ang)
    _, _, truth_wrong = independent_truth(-ang)   # wrong-sign truth

    random.seed(SEED)
    _, impl = apply_random_rotation(canvas_s, pmask_s, mag, mag, C_PAD)

    t_impl, t_truth, t_wrong = token_ignore(impl), token_ignore(truth), token_ignore(truth_wrong)
    L = t_impl.numel()
    gray = (~t_impl & t_truth).sum().item() / L      # pad, but marked content
    drop = (t_impl & ~t_truth).sum().item() / L      # content, but marked pad
    disagree_wrong = (t_impl != t_wrong).sum().item()

    # old behaviour: static mask against the same independent truth
    t_static = token_ignore(pmask_s)
    old_gray = (~t_static & t_truth).sum().item() / L
    old_drop = (t_static & ~t_truth).sum().item() / L

    print(f"    angle {ang:+5.1f}deg  NEW gray={gray*100:5.2f}% drop={drop*100:5.2f}%"
          f"   OLD gray={old_gray*100:5.2f}% drop={old_drop*100:5.2f}%"
          f"   wrong-sign delta={disagree_wrong} tok")
    check(gray == 0.0 and drop == 0.0, f"token-exact vs independent truth @{ang:+.1f}deg",
          f"gray={gray*100:.3f}% drop={drop*100:.3f}%")
    check(disagree_wrong > 0,
          f"test is sign-discriminating @{ang:+.1f}deg (wrong sign would fail)",
          f"wrong-sign truth differs by only {disagree_wrong} tokens")
    check(old_drop > 0.004,
          f"old static mask dropped real content @{ang:+.1f}deg (fix is not a no-op)",
          f"old_drop={old_drop*100:.2f}%")

# ---------------------------------------------------------------- 4
print("\n[4/6] determinism and RNG draw count")
c, pm = process_image_cpu(make_portrait(), TARGET, PAD)
random.seed(1234)
r1, m1 = apply_random_rotation(c, pm, 2.0, 5.0, PAD)
random.seed(1234)
r2, m2 = apply_random_rotation(c, pm, 2.0, 5.0, PAD)
check(torch.equal(m1, m2) and np.array_equal(np.array(r1), np.array(r2)),
      "same seed -> identical canvas and mask")

random.seed(99)
apply_random_rotation(c, pm, 2.0, 5.0, PAD)
after_impl = random.random()
for n_draws, label in ((1, "1 draw"), (2, "2 draws"), (3, "3 draws")):
    random.seed(99)
    random.uniform(2.0, 5.0)
    for _ in range(n_draws - 1):
        random.random()
    ref = random.random()
    if n_draws == 2:
        check(after_impl == ref, "RNG stream advanced by exactly 2 draws")
    else:
        check(after_impl != ref, f"RNG stream is NOT {label} (draw count pinned)")

# ---------------------------------------------------------------- 5
print("\n[5/6] angle sampling drives the produced mask, over many seeds")
LO, HI = 3.0, 6.0
signs, mismatches, oob = [], 0, 0
for seed in range(60):
    ang = expected_angle(seed, LO, HI)
    if not (LO - 1e-9 <= abs(ang) <= HI + 1e-9):
        oob += 1
    signs.append(ang < 0)
    canvas_s, pmask_s, truth = independent_truth(ang)
    random.seed(seed)
    _, impl = apply_random_rotation(canvas_s, pmask_s, LO, HI, C_PAD)
    if not torch.equal(token_ignore(impl), token_ignore(truth)):
        mismatches += 1
check(oob == 0, "|angle| within [min,max] for every seed", f"{oob} out of range")
check(mismatches == 0,
      "produced mask matches the truth for the drawn angle (60 seeds)",
      f"{mismatches} seeds mismatched")
neg = float(np.mean(signs))
check(0.30 < neg < 0.70, "sign roughly balanced", f"neg_frac={neg:.2f}")

# ---------------------------------------------------------------- 6
print("\n[6/6] end-to-end through SidecarJsonDataset (rotation ON)")
with tempfile.TemporaryDirectory() as td:
    root = Path(td)
    vocab = TagVocabulary()
    for tag in ["solo", "tag_b"]:
        idx = len(vocab.tag_to_index)
        vocab.tag_to_index[tag] = idx
        vocab.index_to_tag[idx] = tag
    vocab._ensure_rating_tags()
    vocab.tags = list(vocab.tag_to_index.keys())

    n = 12
    for i in range(n):
        make_portrait().save(root / f"img_{i}.jpg", quality=95)
        (root / f"img_{i}.json").write_text(json.dumps(
            {"filename": f"img_{i}.jpg", "tags": "solo", "rating": "g"}),
            encoding="utf-8")
    try:
        ds = SidecarJsonDataset(
            root_dir=root,
            json_files=sorted(root.glob("img_*.json")),
            vocab=vocab,
            image_size=TARGET,
            pad_color=PAD,
            random_flip_prob=0.0,
            color_jitter_enabled=False,
            random_erasing_enabled=False,
            gaussian_blur_enabled=False,
            metadata_cache_enabled=False,
            prebuilt_arrow_table=None,
            random_rotation_enabled=True,
            random_rotation_p=1.0,
            random_rotation_min_degrees=2.0,
            random_rotation_max_degrees=5.0,
        )
        # Read the dataset's real normalization rather than hardcoding it.
        nm = np.asarray(ds.normalize_mean, dtype=np.float32)
        nsd = np.asarray(ds.normalize_std, dtype=np.float32)
        pv = (np.asarray(PAD, dtype=np.float32) / 255.0 - nm) / nsd

        bad, checked, rotated_seen = 0, 0, 0
        for i in range(min(n, len(ds))):
            s = ds[i]
            pmk, imt = s.get("padding_mask"), s.get("images")
            if pmk is None or imt is None or s.get("error"):
                continue
            checked += 1
            pmk = pmk.squeeze().bool()
            arr = imt.float().numpy()
            if arr.ndim == 3 and arr.shape[0] == 3:
                arr = np.transpose(arr, (1, 2, 0))
            m = pmk.numpy()
            if m.sum() == 0:
                continue
            # A static (unrotated) mask here would be exactly rectangular; the
            # rotated one is not. Verifies the fix is live through the dataset.
            rows = m.all(axis=1)
            if not np.array_equal(m, np.repeat(rows[:, None], m.shape[1], axis=1)):
                rotated_seen += 1
            if np.abs(arr[m] - pv).max() > 0.05:
                bad += 1
        if checked == 0:
            check(False, "dataset yielded samples with images+padding_mask")
        else:
            check(bad == 0,
                  f"masked-pad pixels match pad_color end-to-end ({checked} samples)",
                  f"{bad} samples deviated")
            check(rotated_seen == checked,
                  f"mask is non-rectangular for every sample (rotation reached the mask)",
                  f"only {rotated_seen}/{checked} were non-rectangular")
    except Exception as e:
        check(False, "end-to-end dataset construction", f"{type(e).__name__}: {e}")

print("\n" + "=" * 64)
if FAILURES:
    print(f"FAILED ({len(FAILURES)}): " + "; ".join(FAILURES))
    sys.exit(1)
print("ALL PASS")
