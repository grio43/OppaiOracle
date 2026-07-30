"""
test_flip_pipeline.py

Verify horizontal flip augmentation actually fires through SidecarJsonDataset
and DataLoader workers, after removing the orientation_handler.

Tests:
  1. _should_flip distribution + determinism (no handler dependency)
  2. Pickle round-trip preserves flip state (worker spawn pathway)
  3. End-to-end DataLoader(num_workers>0): ~50% flip rate AND pixel layout
     agrees with the flip_applied flag for every sample.
  4. set_epoch() reaches workers that were ALREADY spawned (persistent_workers),
     which is the order train_direct.py actually uses.

Run from repo root with the project's Python:
    L:/Dab/payton_env/Scripts/python.exe test_flip_pipeline.py
"""

import json
import logging
import multiprocessing as mp
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))
from dataset_loader import SidecarJsonDataset
from vocabulary import TagVocabulary


SIZE = 64
MARKER_COL = 6  # bar near LEFT edge → after flip, ends up near RIGHT edge


def _make_marker_image() -> Image.Image:
    arr = np.full((SIZE, SIZE, 3), 255, dtype=np.uint8)
    arr[:, MARKER_COL - 2: MARKER_COL + 3, :] = 0
    return Image.fromarray(arr)


def _make_dataset(root: Path, n: int) -> None:
    img = _make_marker_image()
    for i in range(n):
        stem = f"img_{i:06d}"
        img.save(root / f"{stem}.png")
        (root / f"{stem}.json").write_text(json.dumps({
            "filename": f"{stem}.png",
            "tags": "tag_a tag_b",
            "rating": "general",
        }), encoding="utf-8")


def _build_vocab() -> TagVocabulary:
    v = TagVocabulary()
    for tag in ["tag_a", "tag_b"]:
        idx = len(v.tag_to_index)
        v.tag_to_index[tag] = idx
        v.index_to_tag[idx] = tag
    v._ensure_rating_tags()
    v.tags = list(v.tag_to_index.keys())
    return v


def _build_dataset(root: Path, vocab: TagVocabulary, *, flip_prob: float = 0.5) -> SidecarJsonDataset:
    json_files = sorted(root.glob("img_*.json"))
    return SidecarJsonDataset(
        root_dir=root,
        json_files=json_files,
        vocab=vocab,
        image_size=SIZE,
        random_flip_prob=flip_prob,
        color_jitter_enabled=False,
        random_erasing_enabled=False,
        random_rotation_enabled=False,
        gaussian_blur_enabled=False,
        metadata_cache_enabled=False,
        prebuilt_arrow_table=None,
    )


def _marker_on_left(img: torch.Tensor) -> bool:
    """img: CHW tensor, normalized to [-1, 1] (mean=0.5, std=0.5).
    The marker is dark (-1.0); white background is +1.0. Returns True iff the
    darkest column lives in the left half of the image.

    Color-order-agnostic: mean=0.5/std=0.5 and pad_color=(114,114,114) are
    channel-symmetric, so this assertion behaves identically under RGB and
    BGR ``color_order`` configurations.
    """
    col_means = img.float().mean(dim=(0, 1)).numpy()
    dark_col = int(np.argmin(col_means))
    return dark_col < (SIZE // 2)


def _stub_dataset_for_decide_flip(p: float = 0.5) -> SidecarJsonDataset:
    ds = SidecarJsonDataset.__new__(SidecarJsonDataset)
    ds.random_flip_prob = p
    ds._current_epoch = mp.Value("i", 0, lock=False)
    ds._epoch_was_set = True
    ds._epoch_warning_issued = False
    ds.logger = logging.getLogger("flip-test")
    return ds


def test_flip_decision_distribution() -> None:
    ds = _stub_dataset_for_decide_flip(0.5)
    modes = [ds._should_flip(f"id_{i:08d}") for i in range(20_000)]
    rate = sum(modes) / len(modes)
    assert 0.47 < rate < 0.53, f"flip rate out of band: {rate:.3f}"
    print(f"PASS  flip-decision distribution: {rate:.3f} at p=0.5")


def test_flip_decision_determinism() -> None:
    ds = _stub_dataset_for_decide_flip(0.5)
    a = [ds._should_flip(f"id_{i}") for i in range(500)]
    b = [ds._should_flip(f"id_{i}") for i in range(500)]
    assert a == b, "decisions not deterministic within a single epoch"

    # Same id should reach BOTH "random" and "none" across enough epochs.
    ids_that_varied = 0
    sample_ids = [f"id_{i}" for i in range(500)]
    for image_id in sample_ids:
        seen = set()
        for ep in range(8):
            ds._current_epoch.value = ep
            seen.add(ds._should_flip(image_id))
            if len(seen) > 1:
                ids_that_varied += 1
                break
    assert ids_that_varied > 400, (
        f"epoch should diversify flip decisions per id; only {ids_that_varied}/500 ids "
        f"saw multiple outcomes across 8 epochs"
    )
    print(f"PASS  flip-decision determinism ({ids_that_varied}/500 ids varied across 8 epochs)")


def test_flip_epoch_independence() -> None:
    """Guard against an affine seed hash collapsing flips to a few global states.

    The previous seed was crc32(f"{id}|epoch{n}"). crc32 is affine over GF(2),
    so the epoch->epoch delta was a single constant XOR applied to every image
    at once: at p=0.5 epochs 0 and 1 produced the IDENTICAL flip set and epochs
    2,3 produced its exact complement. The "each id varies across 8 epochs"
    assertion above passes perfectly under that degenerate schedule, so it
    cannot catch this. These assertions can.

    For genuinely independent per-(id, epoch) coins, two distinct epochs agree
    with probability p^2 + (1-p)^2, and a run of E epochs should expose ~all
    distinct flip sets.
    """
    n = 20_000
    ids = [f"id_{i:08d}" for i in range(n)]
    for p in (0.5, 0.35):
        ds = _stub_dataset_for_decide_flip(p)
        expected_agreement = p * p + (1 - p) ** 2
        sets = []
        for ep in range(8):
            ds._current_epoch.value = ep
            sets.append([ds._should_flip(i) for i in ids])

        # 1. No two epochs may be identical or exactly complementary.
        for a in range(8):
            for b in range(a + 1, 8):
                agree = sum(x == y for x, y in zip(sets[a], sets[b])) / n
                assert abs(agree - expected_agreement) < 0.03, (
                    f"epochs {a},{b} at p={p}: agreement {agree:.4f} vs independent "
                    f"expectation {expected_agreement:.4f} — the seed hash is not "
                    f"mixing the epoch (affine-hash regression?)"
                )
        # 2. Every epoch must be a distinct flip set.
        assert len({tuple(s) for s in sets}) == 8, (
            f"at p={p}, 8 epochs produced fewer than 8 distinct flip sets"
        )
    # 3. A 2-epoch run (FixRes / short refit) must still expose both orientations
    #    to ~2p(1-p) of the corpus. The affine hash gave exactly 0 here.
    ds = _stub_dataset_for_decide_flip(0.5)
    both = 0
    for image_id in ids:
        seen = set()
        for ep in (0, 1):
            ds._current_epoch.value = ep
            seen.add(ds._should_flip(image_id))
        both += len(seen) > 1
    frac = both / n
    assert 0.45 < frac < 0.55, (
        f"2-epoch run exposed both orientations to only {frac:.3f} of ids "
        f"(expected ~0.50); short refits would get no flip augmentation"
    )
    print(f"PASS  flip epoch-independence (2-epoch both-orientation coverage {frac:.3f})")


def test_getstate_setstate_preserves_flip(root: Path) -> None:
    """Worker spawn calls __getstate__ on the main side and __setstate__ on
    the worker side. Verify both halves preserve the flip-decision state and
    no longer drop an orientation_handler attribute."""
    vocab = _build_vocab()
    ds = _build_dataset(root, vocab, flip_prob=0.5)

    state = ds.__getstate__()
    assert "orientation_handler" not in state, (
        "state should not contain orientation_handler — the original bug "
        "was that __getstate__ scrubbed it to None and __setstate__ never "
        "rebuilt it, causing all worker flips to silently no-op."
    )
    assert state.get("random_flip_prob") == 0.5
    assert "flip_overrides_path" not in state
    assert "_force_flip_ids" not in state, (
        "the force_flip/never_flip override machinery was removed with the "
        "direction-sensitive-tag design; it must not come back"
    )

    ds2 = SidecarJsonDataset.__new__(SidecarJsonDataset)
    ds2.__setstate__(state)
    assert ds2.random_flip_prob == 0.5
    # Functional check: decision logic still works after round-trip.
    ds2.set_epoch(0)
    assert isinstance(ds2._should_flip("img_000000"), bool)
    print("PASS  __getstate__/__setstate__ preserves flip state")


def test_dataloader_workers_apply_flips(root: Path) -> None:
    vocab = _build_vocab()
    ds = _build_dataset(root, vocab, flip_prob=0.5)
    ds.set_epoch(7)

    loader = DataLoader(
        ds,
        batch_size=8,
        num_workers=2,
        shuffle=False,
        persistent_workers=False,
    )

    n_total = 0
    n_flag_true = 0
    n_pixel_flipped = 0
    n_inconsistent = 0

    for batch in loader:
        imgs = batch["images"]
        flags = batch["flip_applied"]
        if isinstance(flags, torch.Tensor):
            flags = flags.tolist()
        for img, flag in zip(imgs, flags):
            n_total += 1
            marker_left = _marker_on_left(img)
            n_flag_true += int(bool(flag))
            n_pixel_flipped += int(not marker_left)
            # flag True ↔ marker should be on RIGHT (not left)
            if bool(flag) != (not marker_left):
                n_inconsistent += 1

    flag_rate = n_flag_true / n_total
    pixel_rate = n_pixel_flipped / n_total

    assert n_total >= 32, f"too few samples ({n_total})"
    assert 0.30 < flag_rate < 0.70, f"flip flag rate out of band: {flag_rate:.3f}"
    assert n_inconsistent == 0, (
        f"flip_applied flag and actual pixel layout disagree on "
        f"{n_inconsistent}/{n_total} samples — that's the original bug."
    )
    print(
        f"PASS  DataLoader workers apply flips: {n_total} samples, "
        f"flip_applied={flag_rate:.3f}, pixel-flipped={pixel_rate:.3f}, "
        f"flag/pixel agreement = {n_total - n_inconsistent}/{n_total}"
    )


def test_set_epoch_reaches_live_workers(root: Path) -> None:
    """set_epoch() must reach workers that were ALREADY spawned.

    The test above calls set_epoch() before the loader ever iterates, so the
    value is baked into the dataset pickled at spawn time — that passes even if
    the epoch were a plain int. Training does the opposite order: set_epoch(0) at
    startup, workers spawn on the first iter(), then set_epoch(1..N-1) with
    persistent_workers=True keeping the same processes alive. If the shared
    mp.Value did not survive spawn, every epoch would silently reuse the epoch-0
    flip set and the flip rate would still look correct at ~0.5.
    """
    vocab = _build_vocab()
    ds = _build_dataset(root, vocab, flip_prob=0.5)
    ds.set_epoch(0)

    loader = DataLoader(
        ds,
        batch_size=8,
        num_workers=2,
        shuffle=False,
        persistent_workers=True,
        prefetch_factor=2,
    )

    try:
        sets = {}
        for epoch in range(4):
            ds.set_epoch(epoch)  # AFTER the workers exist (from epoch 1 onward)
            flags = []
            for batch in loader:
                f = batch["flip_applied"]
                f = f.tolist() if isinstance(f, torch.Tensor) else list(f)
                for img, flag in zip(batch["images"], f):
                    assert bool(flag) == (not _marker_on_left(img)), (
                        "flip_applied disagrees with pixel layout"
                    )
                flags.extend(bool(x) for x in f)
            sets[epoch] = tuple(flags)

        distinct = len(set(sets.values()))
        assert distinct == 4, (
            f"4 epochs produced only {distinct} distinct flip sets — set_epoch() is "
            f"not reaching the live persistent workers, so every epoch reuses the "
            f"epoch-0 flips"
        )
    finally:
        del loader

    print(f"PASS  set_epoch() reaches already-spawned persistent workers "
          f"({distinct}/4 distinct flip sets)")


def main() -> None:
    test_flip_decision_distribution()
    test_flip_decision_determinism()
    test_flip_epoch_independence()

    with tempfile.TemporaryDirectory(prefix="oo_flip_test_") as td:
        root = Path(td) / "dataset"
        root.mkdir(parents=True)
        _make_dataset(root, n=64)

        test_getstate_setstate_preserves_flip(root)
        test_dataloader_workers_apply_flips(root)
        test_set_epoch_reaches_live_workers(root)

    print("\nAll flip-pipeline tests passed.")


if __name__ == "__main__":
    main()
