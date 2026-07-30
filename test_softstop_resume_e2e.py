"""END-TO-END integration test for the soft-stop / resume path.

Run directly:  python test_softstop_resume_e2e.py

WHY THIS FILE EXISTS
--------------------
`test_softstop_resume.py` covers the same machinery with 33 STRUCTURAL checks: AST
proofs, the LR-schedule guard block lifted out and exec'd against a stub namespace,
`retarget()` driven directly, and `ResumableSampler` + `DataLoader` exercised in
isolation. Every one of those runs without launching training.

What none of them execute is the INTEGRATION. Two of the six bugs found in this work
were seam bugs -- defects that live in the handoff between components, where each
component is individually correct. A unit test cannot see them by construction. So
this file deliberately stubs NOTHING on the path under test:

  * the real `train_with_orientation_tracking(config)` drives everything
  * the real `create_dataloaders()` builds a real Arrow metadata cache, a real
    train/val split and a real `ResumableSampler` over a real on-disk dataset
  * the real `CheckpointManager.save_checkpoint` writes the checkpoint (through its
    real async writer) and the real `load_checkpoint` reads it back
  * the real LR-schedule guard evaluates genuinely-restored scheduler state
  * the real STOP_TRAINING sentinel poll, latch, accumulation-window handling and
    save-and-exit path run inside the real epoch loop

The only thing added is OBSERVATION: a recording wrapper around the training
dataset's `__getitem__` (the instrumentation approach the seam under test cannot
fake, because sample delivery is precisely what is being asserted). Nothing on the
save/load/guard/sampler path is replaced.

WHY CUDA AND NOT CPU
--------------------
A CPU run is not merely slow here, it is IMPOSSIBLE without editing production code,
and that is itself a finding (reported by this file as a skip reason if CUDA is
absent):

  * `dataset_loader.py` hard-codes the image tensor dtype to bfloat16
    (`self._image_dtype = torch.bfloat16`, ~line 1762, and the `ToDtype(torch.bfloat16)`
    transform ~line 1801). There is no config knob for it.
  * `train_direct.py` (~line 863) RAISES if `training.use_amp` is true and the device
    is not CUDA, so autocast -- the only thing that reconciles bf16 activations with
    fp32 conv/linear weights -- cannot be enabled on CPU.

So on CPU the loader hands bf16 images to fp32 weights with no autocast, and the very
first `conv2d` dies with "Input type (struct c10::BFloat16) and bias type (float)
should be the same". The two constraints are mutually exclusive. This test therefore
runs the production path: CUDA + bf16 AMP.

HERMETICITY
-----------
The test writes ONLY inside its own scratch run directory. Two caches are normally
anchored to the repo root rather than to any config key --
`dataset_loader._PROJ_ROOT/logs/splits` and
`utils.metadata_cache._PROJ_ROOT/logs/metadata_cache` -- so the child process
redirects both module attributes into the scratch dir before training starts. The
parent additionally sweeps the repo-side cache keys for its own dataset path on exit,
in case that redirection ever stops working. The real experiment directories
(experiments/run1_vit, experiments/run2_vit) and the real dataset (L:/Dab/Dab) are
never read or written.
"""
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))

# Scratch root. Kept off the project tree so a crashed run cannot leave debris in the
# repo; recreated from scratch on every invocation so the test is re-runnable with no
# manual cleanup.
SCRATCH = Path(
    os.environ.get(
        "OO_E2E_SCRATCH",
        r"Z:\claude_temp\claude\l--Dab-OppaiOracle\5ea49fbe-a034-449a-86cd-e59e70fccb69\scratchpad\e2e_run",
    )
)

# ---------------------------------------------------------------- toy-scale shape
#
# These numbers are chosen, not arbitrary. See `test_resume_across_a_batch_size_growth`
# for why NEW_BATCH must exceed the recorded sample offset.
N_IMAGES = 200          # -> 190 train / 10 val at the loader's hardcoded 0.95 split
IMAGE_SIZE = 64         # divisible by PATCH
PATCH = 16
BATCH = 4
NEW_BATCH = 16          # > the sample_in_epoch recorded at STOP_AFTER_ITEMS (see below)
ACCUM = 3               # >1 so the accumulation-window interaction is under test
EPOCHS = 2
SEED = 1234

# Fire STOP_TRAINING once this many training samples have been served. 4 == one
# batch, which makes the stop land in the first accumulation window of epoch 0 and
# keeps the recorded sample offset small enough for NEW_BATCH to exceed it.
STOP_AFTER_ITEMS = 4

# Realistic danbooru-style tags. NOT "tag_00" style: `vocabulary.py`'s integrity check
# refuses a vocabulary containing more than 10 `tag_\d+` placeholder names, so a
# synthetic vocabulary built the obvious way is rejected outright.
TAGS = [
    "1girl", "solo", "long_hair", "smile", "blue_eyes", "blush",
    "open_mouth", "short_hair", "simple_background", "shirt",
    "looking_at_viewer", "hat",
]

_FAILURES = []
_CHECKS = [0]


def check(cond, msg):
    _CHECKS[0] += 1
    if cond:
        print(f"PASS: {msg}")
    else:
        print(f"FAIL: {msg}")
        _FAILURES.append(msg)


# ============================================================================ fixture


def build_fixture(run_dir: Path):
    """Synthesize a tiny sidecar-mode dataset + vocabulary the real loader accepts.

    Layout is per-image JSON sidecars next to their images, which is what
    `create_dataloaders` falls into when the directory has no train.json/val.json/images/
    manifest triple. Each sidecar needs `filename`, `tags` and a RECOGNISED `rating` --
    an unknown rating makes the Arrow keep-mask drop the row entirely, which would
    silently shrink the dataset instead of failing.
    """
    from PIL import Image
    import numpy as np

    data = run_dir / "data"
    data.mkdir(parents=True, exist_ok=True)
    for i in range(N_IMAGES):
        stem = f"img{i:05d}"
        arr = (np.random.RandomState(i).rand(48, 48, 3) * 255).astype("uint8")
        Image.fromarray(arr).save(data / f"{stem}.png")
        tags = [TAGS[(i + k) % len(TAGS)] for k in range(3)]
        (data / f"{stem}.json").write_text(
            json.dumps({"filename": f"{stem}.png", "tags": tags, "rating": "general"}),
            encoding="utf-8",
        )

    # Minimal vocabulary. Indices must be gapless 0..N-1 and the two maps must be exact
    # inverses, or `verify_vocabulary_integrity` rejects the file.
    all_tags = ["<PAD>", "<UNK>"] + TAGS + [
        "rating:general", "rating:sensitive", "rating:questionable", "rating:explicit",
    ]
    vocab = {"tag_to_index": {}, "index_to_tag": {}, "tag_frequencies": {}}
    for i, t in enumerate(all_tags):
        vocab["tag_to_index"][t] = i
        vocab["index_to_tag"][str(i)] = t
        if not t.startswith("<"):
            vocab["tag_frequencies"][t] = 100
    (run_dir / "vocabulary.json").write_text(json.dumps(vocab), encoding="utf-8")
    return data


# =========================================================================== child


def build_config(run_dir: Path, spec: dict):
    """Build the toy FullConfig.

    Loaded from the shipped unified_config.yaml so the real YAML parsing / defaults /
    cross-field syncing all run, then overridden in memory. The YAML on disk is never
    modified, and `output_root` / `log_dir` / `vocab_path` are all redirected into the
    scratch dir so the real experiment directories are untouchable from here.
    """
    from Configuration_System import load_config

    cfg = load_config(str(REPO / "configs" / "unified_config.yaml"), validate=False)

    cfg.experiment_name = "oo_e2e_vit"   # already carries the arch token -> stable
    cfg.output_root = str(run_dir / "experiments")
    cfg.log_dir = str(run_dir / "logs")
    cfg.vocab_path = str(run_dir / "vocabulary.json")
    cfg.file_logging_enabled = False

    cfg.data.storage_locations = [
        {"path": str(run_dir / "data"), "priority": 0, "type": "local", "enabled": True}
    ]
    cfg.data.image_size = IMAGE_SIZE
    cfg.model.image_size = IMAGE_SIZE
    cfg.model.patch_size = PATCH
    cfg.data.patch_size = PATCH
    cfg.model.hidden_size = 32
    cfg.model.num_hidden_layers = 2
    cfg.model.num_attention_heads = 2
    cfg.model.intermediate_size = 64
    cfg.model.num_labels = 0             # auto-detect from the toy vocabulary
    cfg.model.use_flex_attention = False
    cfg.model.gradient_checkpointing = False

    cfg.data.batch_size = spec.get("batch_size", BATCH)
    # num_workers=0 keeps the dataset in-process so the __getitem__ recorder sees every
    # served sample directly. It also sidesteps Windows spawn, which cannot pickle the
    # mp.Queue train_direct attaches to config.data.stats_queue.
    cfg.data.num_workers = 0
    cfg.data.persistent_workers = False
    cfg.data.pin_memory = False
    cfg.data.random_flip_prob = 0.0
    cfg.validation.dataloader.batch_size = 4
    cfg.validation.dataloader.num_workers = 0
    cfg.validation.dataloader.persistent_workers = False
    cfg.validation.dataloader.pin_memory = False

    cfg.training.device = "cuda"
    cfg.training.use_amp = True          # see module docstring: CPU is not an option
    cfg.training.use_compile = False
    cfg.training.num_epochs = EPOCHS
    cfg.training.warmup_epochs = 1
    cfg.training.gradient_accumulation_steps = ACCUM
    cfg.training.eval_steps = 10 ** 9    # keep epoch-end validation off the hot path
    cfg.training.logging_steps = 10 ** 9
    cfg.training.save_steps = 10 ** 9    # only the SOFT STOP may write a checkpoint
    cfg.training.save_best_only = False  # else the numbered soft-stop file is skipped
    cfg.training.seed = SEED
    cfg.training.use_tensorboard = False
    cfg.training.resume_from = spec.get("resume_from", "none")
    cfg.training.on_lr_schedule_change = spec.get("policy", "error")

    cfg.monitor.track_system_metrics = False
    cfg.monitor.enable_alerts = False
    cfg.monitor.log_to_file = False

    Path(cfg.log_dir).mkdir(parents=True, exist_ok=True)
    cfg.validate()
    return cfg


def run_child(spec: dict):
    """The training subprocess.

    A fresh process per phase, which is also how a real stop/resume works: the operator
    stops a run and starts a new one. It additionally guarantees no module-level state
    leaks between phases and makes the stale-sentinel unlink genuinely stale.
    """
    import torch  # the consumed-samples counter's grad-enabled guard needs it here

    run_dir = Path(spec["run_dir"])
    out_path = Path(spec["out"])
    result = {"phase": spec["phase"], "error": None, "returned": False,
              "events": [], "sched": None, "epoch_marks": []}

    os.environ["OO_NON_INTERACTIVE"] = "1"
    os.environ["OO_AUTO_REBUILD_VOCAB"] = "0"   # never regenerate the repo vocabulary

    # Redirect the two repo-anchored caches into the scratch dir. Both modules read
    # `_PROJ_ROOT` inside the functions that build the paths, so rebinding the module
    # attribute before any dataloader is constructed is sufficient.
    import dataset_loader
    import utils.metadata_cache as _mc
    dataset_loader._PROJ_ROOT = run_dir
    _mc._PROJ_ROOT = run_dir

    import train_direct
    from dataset_loader import create_dataloaders as _real_create_dataloaders

    cfg = build_config(run_dir, spec)

    stop_after = int(spec.get("stop_after_items", 0))
    sentinel = Path(cfg.log_dir) / "STOP_TRAINING"
    captured = {"scheduler": None}
    events = result["events"]

    # -- capture the REAL scheduler instance -----------------------------------------
    # The LR-schedule guard mutates (or refuses to mutate) this exact object. Holding a
    # reference lets the assertions read the post-guard geometry off the live scheduler
    # rather than off a reconstruction, so `rescale` vs `keep` is observed, not inferred.
    _RealSched = train_direct.CosineAnnealingWarmupRestarts

    class _CapturingScheduler(_RealSched):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            captured["scheduler"] = self

    train_direct.CosineAnnealingWarmupRestarts = _CapturingScheduler

    # -- count every sample the epoch loop actually TRAINS ON -------------------------
    # The served-index ledger below CANNOT observe the original double-skip. The pre-fix
    # fallback discarded batches with `next(train_iter)` (train_direct.py ~1902) AFTER
    # the dataset had already served them, so __getitem__ coverage is byte-identical
    # with and without the bug. Mutation testing proved exactly that: reverting the gate
    # to `start_step == 0` dropped the replayed epoch from 178 trained samples to 130
    # while every served-index assertion still reported PASS.
    #
    # The criterion is downstream of the discard, so it only ever sees microbatches the
    # loop actually trained on - the same quantity train_direct accumulates into
    # total_train_samples. torch.is_grad_enabled() separates training from validation,
    # which calls this same object under no_grad (train_direct.py:2747).
    _RealLoss = train_direct.MultiTaskLoss
    consumed = {"n": 0}

    class _CountingLoss(_RealLoss):
        def forward(self, tag_logits, tag_targets, *a, **kw):
            if torch.is_grad_enabled():
                consumed["n"] += int(tag_targets.shape[0])
            return super().forward(tag_logits, tag_targets, *a, **kw)

    train_direct.MultiTaskLoss = _CountingLoss

    # -- record every sample the training dataset actually serves ---------------------
    def _patched_create_dataloaders(**kw):
        train_loader, val_loader, vocab = _real_create_dataloaders(**kw)
        ds = train_loader.dataset
        cls = type(ds)
        orig_getitem = cls.__getitem__
        orig_set_epoch = getattr(cls, "set_epoch", None)
        counter = {"n": 0}

        def hooked_getitem(self, idx):
            out = orig_getitem(self, idx)
            # The val dataset is a different INSTANCE of the same class, so the patch is
            # identity-guarded; without this, validation samples would pollute the
            # served-index ledger that assertion #4 depends on.
            if self is not ds:
                return out
            counter["n"] += 1
            events.append({"ev": "item", "n": counter["n"], "idx": int(idx)})
            if stop_after and counter["n"] == stop_after:
                # Created HERE, not before launch: train_direct unlinks a stale
                # STOP_TRAINING early in startup (~line 976) so a previous run's
                # sentinel cannot immediately re-stop the new one. Firing it from
                # inside a served sample is deterministic -- at num_workers=0 the
                # dataset is only ever touched from the epoch loop, which is strictly
                # after that unlink -- and needs no sleeps or polling threads.
                sentinel.write_text("e2e", encoding="utf-8")
                events.append({"ev": "sentinel", "n": counter["n"]})
            return out

        def hooked_set_epoch(self, epoch):
            if orig_set_epoch is not None:
                orig_set_epoch(self, epoch)
            if self is not ds:
                return
            sched = captured["scheduler"]
            # Snapshot the scheduler at each epoch start. The FIRST snapshot of a
            # resumed run is taken after load_checkpoint + the LR-schedule guard have
            # run but before any optimizer step, i.e. it is exactly the restored state.
            mark = {"ev": "epoch", "epoch": int(epoch), "n": counter["n"],
                    "consumed": consumed["n"]}
            if sched is not None:
                mark["step_in_cycle"] = float(getattr(sched, "step_in_cycle", -1))
                try:
                    mark["lr"] = float(sched.optimizer.param_groups[0]["lr"])
                except Exception:
                    mark["lr"] = None
            events.append(mark)
            result["epoch_marks"].append(mark)

        cls.__getitem__ = hooked_getitem
        if orig_set_epoch is not None:
            cls.set_epoch = hooked_set_epoch
        return train_loader, val_loader, vocab

    train_direct.create_dataloaders = _patched_create_dataloaders

    try:
        train_direct.train_with_orientation_tracking(cfg)
        result["returned"] = True
    except BaseException as exc:  # noqa: BLE001 - the guard's RuntimeError is a result
        result["error"] = f"{type(exc).__name__}: {exc}"

    result["consumed_total"] = consumed["n"]
    sched = captured["scheduler"]
    if sched is not None:
        result["sched"] = {
            "base_max_lr": float(getattr(sched, "base_max_lr", float("nan"))),
            "warmup_steps": int(getattr(sched, "warmup_steps", -1)),
            "first_cycle_steps": int(getattr(sched, "first_cycle_steps", -1)),
            "cur_cycle_steps": int(getattr(sched, "cur_cycle_steps", -1)),
            "min_lr": float(getattr(sched, "min_lr", float("nan"))),
            "gamma": float(getattr(sched, "gamma", float("nan"))),
            "step_in_cycle": float(getattr(sched, "step_in_cycle", -1)),
        }
    out_path.write_text(json.dumps(result), encoding="utf-8")
    return 0


# ========================================================================== parent


def launch(run_dir: Path, phase: str, **spec):
    """Run one training phase in a fresh subprocess and return its result dict."""
    out = run_dir / f"result_{phase}.json"
    if out.exists():
        out.unlink()
    spec.update({"phase": phase, "run_dir": str(run_dir), "out": str(out)})
    spec_path = run_dir / f"spec_{phase}.json"
    spec_path.write_text(json.dumps(spec), encoding="utf-8")

    env = dict(os.environ)
    env["OO_E2E_CHILD"] = str(spec_path)
    env["OO_NON_INTERACTIVE"] = "1"
    env["OO_AUTO_REBUILD_VOCAB"] = "0"
    t0 = time.time()
    proc = subprocess.run(
        [sys.executable, str(Path(__file__).resolve())],
        env=env, cwd=str(REPO), capture_output=True, text=True,
    )
    dt = time.time() - t0
    if not out.exists():
        print(f"  child stdout tail:\n{proc.stdout[-3000:]}")
        print(f"  child stderr tail:\n{proc.stderr[-3000:]}")
        raise RuntimeError(f"phase {phase!r} produced no result (rc={proc.returncode})")
    res = json.loads(out.read_text(encoding="utf-8"))
    print(f"  [phase {phase}] {dt:.0f}s  returned={res['returned']} "
          f"error={(res['error'] or '')[:70]}")
    return res


def served_indices(res):
    return [e["idx"] for e in res["events"] if e["ev"] == "item"]


def _epoch_buckets(res):
    """Bucket the ledger by set_epoch marker -> [(served_indices, consumed_count), ...].

    Both series MUST be derived here, and empty buckets MUST be dropped identically for
    both. train_direct calls dataset.set_epoch() before the epoch starts serving, and a
    resumed run emits an extra marker before its first sample, so the raw bucket list is
    offset from the epochs that actually did work (observed: consumed marks [0, 0, 178]
    for two working epochs). per_epoch_indices() has always dropped the empties; when
    the consumed series was computed separately and did NOT, the two silently indexed
    different epochs and every consumed-vs-served comparison read 0.

    `consumed` is the criterion counter, which sits DOWNSTREAM of the fallback's
    `next(train_iter)  # Consume and discard`. Served-but-not-consumed is therefore the
    double-skip signature, and it is invisible to a served-only ledger.
    """
    marks = [e for e in res["events"] if e["ev"] == "epoch"]
    served = [[] for _ in marks]
    cur = -1
    for e in res["events"]:
        if e["ev"] == "epoch":
            cur += 1
        elif e["ev"] == "item" and cur >= 0:
            served[cur].append(e["idx"])
    bounds = [m.get("consumed", 0) for m in marks] + [res.get("consumed_total", 0)]
    consumed = [bounds[i + 1] - bounds[i] for i in range(len(marks))]
    return [(s, c) for s, c in zip(served, consumed) if s]


def per_epoch_consumed(res):
    """Samples the epoch loop actually TRAINED ON, one count per working epoch."""
    return [c for _, c in _epoch_buckets(res)]


def per_epoch_indices(res):
    """Split the served ledger into one list per working epoch."""
    return [s for s, _ in _epoch_buckets(res)]


def read_ckpt(path: Path):
    import torch
    return torch.load(path, map_location="cpu", weights_only=False)


def snapshot(src: Path, dst: Path):
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def restore(snap: Path, dst: Path):
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(snap, dst)


def tamper_offset(ckpt_path: Path, extra_batches: int):
    """Advance the checkpoint's recorded position by `extra_batches` batches.

    NEGATIVE CONTROL. This forges, in test-owned data, the exact on-disk signature the
    pre-fix double-skip produced: the resume consumes the sampler offset AND then skips
    `resume_batch_idx` further batches, so the replayed epoch begins
    `batch_in_epoch * batch_size` samples later than it should and those samples are
    never trained on in that epoch.

    Reproducing it this way rather than by reverting the fix in train_direct.py keeps
    production code untouched while still proving the coverage assertion has teeth --
    a test that cannot be made to fail is not evidence of anything.
    """
    import torch
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    ts = ck["training_state"]
    lost = extra_batches * BATCH
    ts["batch_in_epoch"] += extra_batches
    ts["sample_in_epoch"] += lost
    ck["training_state"] = ts
    torch.save(ck, ckpt_path)
    return lost


def sweep_repo_caches(data_dir: Path):
    """Belt-and-braces: remove any repo-side cache keyed to our scratch dataset path.

    The child redirects `_PROJ_ROOT`, so these should never exist. If that redirection
    ever breaks, this keeps the repo clean instead of accumulating junk under logs/.
    """
    key = hashlib.sha1(str(data_dir.resolve()).encode("utf-8")).hexdigest()[:16]
    for pattern in (f"logs/splits/{key}.*", f"logs/metadata_cache/{key}.*"):
        for p in REPO.glob(pattern):
            try:
                p.unlink()
                print(f"  swept repo cache leftover: {p.relative_to(REPO)}")
            except OSError:
                pass


# ========================================================================== the test


def main():
    import torch
    if not torch.cuda.is_available():
        # Not a silent skip: state plainly that the CPU path is blocked by production
        # code, so this never reads as "the test passed".
        print("SKIP: CUDA is unavailable. This test cannot fall back to CPU -- "
              "dataset_loader hard-codes bfloat16 image tensors and train_direct "
              "refuses use_amp on a non-CUDA device, so a CPU run cannot reconcile "
              "bf16 activations with fp32 weights. See the module docstring.")
        return 0

    run_dir = SCRATCH
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True)
    ckpt_dir = run_dir / "experiments" / "oo_e2e_vit" / "checkpoints"

    try:
        print(f"Building toy fixture ({N_IMAGES} images) in {run_dir} ...")
        data_dir = build_fixture(run_dir)

        # ---------------------------------------------------------------- phase 1
        print("\n[1/6] fresh run, soft-stopped mid-epoch by a real STOP_TRAINING sentinel")
        p1 = launch(run_dir, "stop", resume_from="none", stop_after_items=STOP_AFTER_ITEMS)
        test_soft_stop_writes_a_mid_epoch_checkpoint(p1, ckpt_dir)

        snap = run_dir / "ckpt_snapshot"
        snapshot(ckpt_dir, snap)
        ck = read_ckpt(ckpt_dir / "last.pt")
        ts = ck["training_state"]
        stop_sample = ts["sample_in_epoch"]
        served1 = served_indices(p1)

        # ---------------------------------------------------------------- phase 2
        print("\n[2/6] resume at the SAME batch size, run to completion")
        p2 = launch(run_dir, "resume_same", resume_from="latest")
        test_resume_lands_at_the_recorded_offset(p1, p2, ck, stop_sample)
        test_no_sample_lost_or_duplicated(served1, p2, "same batch size")

        # ---------------------------------------------------------------- phase 3
        print("\n[3/6] resume with a CHANGED batch size, default policy 'error'")
        restore(snap, ckpt_dir)
        p3 = launch(run_dir, "resume_grow_error", resume_from="latest",
                    batch_size=NEW_BATCH, policy="error")
        test_lr_guard_refuses_by_default(p3)

        # ---------------------------------------------------------------- phase 4
        print("\n[4/6] resume with a CHANGED batch size, policy 'rescale'")
        restore(snap, ckpt_dir)
        p4 = launch(run_dir, "resume_grow_rescale", resume_from="latest",
                    batch_size=NEW_BATCH, policy="rescale")
        test_lr_guard_rescale_retargets(p3, p4)
        test_resume_across_a_batch_size_growth(served1, p4, stop_sample)

        # ---------------------------------------------------------------- phase 5
        print("\n[5/6] resume with a CHANGED batch size, policy 'keep'")
        restore(snap, ckpt_dir)
        p5 = launch(run_dir, "resume_grow_keep", resume_from="latest",
                    batch_size=NEW_BATCH, policy="keep",
                    stop_after_items=STOP_AFTER_ITEMS)
        test_lr_guard_keep_preserves_checkpoint_geometry(ck, p4, p5)

        # ---------------------------------------------------------------- phase 6
        print("\n[6/6] NEGATIVE CONTROL: forge a double-skip and confirm it is caught")
        restore(snap, ckpt_dir)
        forged = tamper_offset(ckpt_dir / "last.pt", extra_batches=ts["batch_in_epoch"])
        p6 = launch(run_dir, "negative_control", resume_from="latest")
        test_coverage_check_detects_a_forged_double_skip(served1, p6, forged)

    finally:
        sweep_repo_caches(run_dir / "data")
        if run_dir.exists() and not os.environ.get("OO_E2E_KEEP"):
            shutil.rmtree(run_dir, ignore_errors=True)
            print(f"\nCleaned up {run_dir}")

    print()
    if _FAILURES:
        print(f"{len(_FAILURES)} OF {_CHECKS[0]} CHECKS FAILED")
        for f in _FAILURES:
            print(f"  - {f}")
        return 1
    print(f"ALL {_CHECKS[0]} E2E CHECKS PASSED")
    return 0


# ------------------------------------------------------------------- assertions


def test_soft_stop_writes_a_mid_epoch_checkpoint(p1, ckpt_dir):
    """#1 A real sentinel must stop the run and leave a MID-EPOCH checkpoint.

    Protects against: the soft stop being unreachable at all (the pre-fix throttled
    check could never coincide with an accumulation boundary for many `accum` values,
    so STOP_TRAINING latched a flag and the run continued forever); and against a stop
    that exits without persisting its position, which would silently replay the whole
    epoch on resume.
    """
    check(p1["returned"] and p1["error"] is None,
          "soft-stopped run exits cleanly (no exception)")
    check((ckpt_dir / "last.pt").exists(),
          "soft stop wrote last.pt")
    numbered = sorted(ckpt_dir.glob("checkpoint_epoch_*_step_*.pt"))
    check(len(numbered) == 1,
          f"soft stop wrote exactly one numbered checkpoint (found {len(numbered)})")

    ck = read_ckpt(ckpt_dir / "last.pt")
    ts = ck["training_state"]
    check(ts["is_epoch_boundary"] is False,
          "checkpoint is flagged mid-epoch (is_epoch_boundary == False)")
    check(ts["batch_in_epoch"] > 0 and ts["sample_in_epoch"] > 0,
          f"checkpoint records a nonzero position "
          f"(batch_in_epoch={ts['batch_in_epoch']}, sample_in_epoch={ts['sample_in_epoch']})")

    # THE consistency assertion: the recorded sample offset must equal the number of
    # samples the dataset ACTUALLY served. Counters are part of what is under test, so
    # this is measured from the dataset side, not from `step`.
    n_served = len(served_indices(p1))
    check(ts["sample_in_epoch"] == n_served,
          f"sample_in_epoch ({ts['sample_in_epoch']}) equals the samples actually "
          f"served ({n_served})")
    check(ts["batch_in_epoch"] * BATCH == n_served,
          f"batch_in_epoch ({ts['batch_in_epoch']}) x batch_size ({BATCH}) equals "
          f"samples served ({n_served})")
    check(ck["step"] == n_served // (BATCH * ACCUM),
          f"global_step ({ck['step']}) equals completed optimizer updates "
          f"({n_served // (BATCH * ACCUM)})")
    # The stop must land inside epoch 0, not at its boundary -- otherwise the mid-epoch
    # resume path this whole file exercises is never taken.
    check(n_served < 190,
          f"the stop happened mid-epoch, not at the epoch boundary ({n_served} < 190)")


def test_resume_lands_at_the_recorded_offset(p1, p2, ck, stop_sample):
    """#2 The replayed epoch must start at the recorded offset and continue the schedule.

    Protects against: resuming at 0 (whole epoch replayed) or double-skipping; and
    against the scheduler/global_step silently restarting, which would re-run warmup
    and retrain at the wrong point on the cosine.
    """
    check(p2["returned"] and p2["error"] is None,
          "resumed run completes without error")

    epochs = per_epoch_indices(p2)
    check(len(epochs) == EPOCHS,
          f"resumed run entered {len(epochs)} epochs (expected {EPOCHS})")

    replayed = epochs[0]
    total_train = len(set(served_indices(p1)) | set(replayed))
    check(len(replayed) == total_train - stop_sample,
          f"replayed epoch served {len(replayed)} samples, expected "
          f"{total_train - stop_sample} (= {total_train} total - {stop_sample} offset)")
    check(len(replayed) != total_train,
          "the replayed epoch did NOT restart from 0")

    # The scheduler snapshot taken at the first epoch start of the resumed run is the
    # restored state, before any optimizer step of this process.
    #
    # NOTE: train_direct calls dataset.set_epoch(0) once during startup (~line 569,
    # right after create_dataloaders) which is BEFORE the scheduler is constructed
    # (~line 845). That first mark therefore carries no scheduler snapshot; the first
    # mark that does is the one from the epoch loop itself.
    marks = [m for m in p2["epoch_marks"] if "step_in_cycle" in m]
    first = marks[0] if marks else {}

    # The scheduler's position is compared against the value the CHECKPOINT saved, not
    # against global_step. `_LRScheduler.__init__` calls step() once, so a freshly
    # constructed scheduler already sits at step_in_cycle == 1 and the standing
    # invariant is `step_in_cycle == global_step + 1`. Asserting equality with
    # global_step would encode an off-by-one as the expectation; asserting equality
    # with the saved value is what "continues off the restored schedule" actually means.
    saved_sic = (ck.get("scheduler_state_dict") or {}).get("step_in_cycle")
    check(saved_sic is not None and first.get("step_in_cycle") == saved_sic,
          f"resumed scheduler continues at the checkpoint's step_in_cycle={saved_sic} "
          f"(observed {first.get('step_in_cycle')})")
    check(saved_sic == ck["step"] + 1,
          f"checkpoint's scheduler position tracks global_step "
          f"(step_in_cycle {saved_sic} == global_step {ck['step']} + 1)")
    # A scheduler that had restarted would read exactly 1 -- the post-construction
    # value -- so this is what separates "restored" from "rebuilt".
    check(first.get("step_in_cycle", 0) > 1,
          f"restored scheduler did NOT restart at the fresh-construction position "
          f"(step_in_cycle={first.get('step_in_cycle')} > 1)")
    check(first.get("lr") is not None and first["lr"] > 0,
          f"resumed run starts with a live LR off the restored schedule "
          f"(lr={first.get('lr')})")


def test_no_sample_lost_or_duplicated(served1, p2, label):
    """#4 Across a full stop -> resume, every dataset index is seen exactly once.

    THE assertion. It is the one no existing test makes against the real loop, and the
    one that would have caught the original double-skip: a resume that consumed the
    sampler offset AND then ran the slow fallback dropped up to new_bs^2/old_bs samples
    from the replayed epoch, and nothing anywhere reported it -- the loss curve simply
    trained on less data than it claimed.

    Measured from the ids the dataset actually served, never from step counters,
    because the counters are themselves under test.

    NOTE the served ledger alone is NOT sufficient: batches can be discarded after being
    served, which leaves it unchanged. The consumed-vs-served check below closes that
    gap - see test_resume_across_a_batch_size_growth for the mutation-tested case.
    """
    replayed = per_epoch_indices(p2)[0]
    before, after = list(served1), list(replayed)
    union = set(before) | set(after)
    overlap = set(before) & set(after)

    check(not overlap,
          f"[{label}] no sample is served twice across the stop/resume boundary "
          f"({len(overlap)} duplicates)")
    check(len(before) + len(after) == len(union),
          f"[{label}] the replayed epoch has no internal duplicates "
          f"({len(before)}+{len(after)} served vs {len(union)} distinct)")
    check(len(union) == max(union) + 1,
          f"[{label}] the stop+resume epoch covers every index 0..{max(union)} "
          f"with no gaps ({len(union)} distinct)")
    consumed = per_epoch_consumed(p2)[0]
    check(consumed == len(after),
          f"[{label}] every served sample was actually trained on "
          f"(consumed {consumed} == served {len(after)})")


def test_resume_across_a_batch_size_growth(served1, p4, stop_sample):
    """#4 (the hard case) No loss/duplication when batch_size GREW past the offset.

    This shape is the exact precondition of the original bug. With
    sample_in_epoch < new batch_size, `start_step = offset // batch_size` floors to 0.
    The pre-fix fallback was gated on `start_step == 0`, so it fired even though the
    sampler offset had ALREADY been applied, and consumed `resume_batch_idx` further
    batches on top of it. Post-fix the gate is `not sampler_offset_applied`.

    The unit suite proves this arithmetically; here it is proven against the real
    sampler, the real DataLoader and the real epoch loop.
    """
    check(NEW_BATCH > stop_sample,
          f"the test shape actually reproduces the bug precondition "
          f"(new batch {NEW_BATCH} > recorded offset {stop_sample}, so start_step floors to 0)")
    check(p4["returned"] and p4["error"] is None,
          "resume at the grown batch size completes without error")

    replayed = per_epoch_indices(p4)[0]
    union = set(served1) | set(replayed)
    overlap = set(served1) & set(replayed)
    check(not overlap,
          f"[grown batch] no duplicate samples across the boundary ({len(overlap)} found)")
    check(len(union) == max(union) + 1 and len(served1) + len(replayed) == len(union),
          f"[grown batch] every index served exactly once "
          f"({len(served1)} + {len(replayed)} = {len(union)} distinct)")

    # Name the pre-fix damage explicitly so a regression reads as a number, not a shape.
    would_have_dropped = (stop_sample // BATCH) * NEW_BATCH
    check(len(replayed) == len(union) - stop_sample,
          f"[grown batch] replayed epoch served all {len(union) - stop_sample} remaining "
          f"samples; the pre-fix double-skip would have dropped a further "
          f"{would_have_dropped}")

    # THE assertion for this bug. Everything above measures what the DATASET SERVED, and
    # the double-skip discards batches after they are served - mutation testing showed
    # all of the above passing while 48 samples were being thrown away. This compares
    # served against what the epoch loop actually TRAINED ON; the pre-fix form makes the
    # two diverge by exactly `would_have_dropped`.
    consumed = per_epoch_consumed(p4)[0]
    check(consumed == len(replayed),
          f"[grown batch] every served sample was actually trained on "
          f"(consumed {consumed} == served {len(replayed)}); the pre-fix double-skip "
          f"discards {would_have_dropped} of them post-serve")


def test_coverage_check_detects_a_forged_double_skip(served1, p6, forged):
    """The coverage assertion must FAIL on a run that really did drop samples.

    Without this, every "no sample lost" PASS above is unfalsifiable: an assertion that
    holds no matter what the run does proves nothing about the run. Here the offset was
    deliberately advanced, so `forged` samples must go missing from the replayed epoch,
    and the same arithmetic that reports success above must report exactly that loss.
    """
    check(p6["returned"] and p6["error"] is None,
          "negative-control run completes (the damage is silent, which is the point)")
    replayed = per_epoch_indices(p6)[0]
    union = set(served1) | set(replayed)
    total = 190
    missing = total - len(union)
    check(missing == forged,
          f"the coverage check detects exactly the {forged} forged-away samples "
          f"(measured {missing} missing)")
    check(len(union) != total,
          "the 'every index exactly once' assertion genuinely fails when samples are lost")


def test_lr_guard_refuses_by_default(p3):
    """#3 The default 'error' policy must refuse a resume whose LR geometry drifted.

    Evaluated against GENUINELY restored scheduler state -- load_checkpoint really put
    the old geometry on the live scheduler -- rather than against a stub namespace.
    Protects against silently continuing at the previous run's calibration.
    """
    err = p3["error"] or ""
    check(not p3["returned"] and err.startswith("RuntimeError"),
          f"changed batch size raises under the default policy (got: {err[:90] or 'no error'})")
    check("effective batch size" in err,
          "the error names the drifted effective-batch dimension")
    # Changing batch_size also moves updates_per_epoch, hence both schedule lengths.
    check("warmup_steps" in err and "first_cycle_steps" in err,
          "the error also names the schedule-length dimensions that moved with it")
    for hint in ("'rescale'", "'keep'"):
        check(hint in err, f"the error offers the {hint} policy")


def test_lr_guard_rescale_retargets(p3, p4):
    """#3 'rescale' must complete AND actually retarget the live scheduler.

    Protects against a policy that logs its intent but leaves the restored geometry in
    place -- the failure mode that makes the run anneal to the wrong floor at the wrong
    peak while reporting that it rescaled.
    """
    check(p4["returned"] and p4["error"] is None,
          "'rescale' completes the resume instead of raising")
    s4 = p4["sched"]
    check(s4 is not None, "captured the live scheduler for the rescale run")
    if not s4:
        return
    # The current config's geometry, recomputed here from the toy shape. At NEW_BATCH
    # the loader yields a different number of batches, so updates_per_epoch -- and with
    # it warmup_steps and first_cycle_steps -- must follow the NEW config.
    upe = (((190 + NEW_BATCH - 1) // NEW_BATCH) // ACCUM)
    check(s4["warmup_steps"] == 1 * upe,
          f"rescale retargeted warmup_steps to the new config's {1 * upe} "
          f"(got {s4['warmup_steps']})")
    check(s4["first_cycle_steps"] == EPOCHS * upe,
          f"rescale retargeted first_cycle_steps to the new config's {EPOCHS * upe} "
          f"(got {s4['first_cycle_steps']})")


def test_lr_guard_keep_preserves_checkpoint_geometry(ck, p4, p5):
    """#3 'keep' must proceed using the CHECKPOINT's schedule, not the config's.

    The contrast with 'rescale' is the point: if both policies produced the same
    scheduler, neither would be doing anything.
    """
    check(p5["returned"] and p5["error"] is None,
          "'keep' completes the resume instead of raising")
    s5, s4 = p5["sched"], p4["sched"]
    check(s5 is not None, "captured the live scheduler for the keep run")
    if not (s5 and s4):
        return
    saved = ck.get("scheduler_state_dict") or {}
    if "warmup_steps" in saved:
        check(s5["warmup_steps"] == saved["warmup_steps"],
              f"'keep' left warmup_steps at the checkpoint's {saved['warmup_steps']} "
              f"(got {s5['warmup_steps']})")
        check(s5["first_cycle_steps"] == saved.get("first_cycle_steps"),
              f"'keep' left first_cycle_steps at the checkpoint's "
              f"{saved.get('first_cycle_steps')} (got {s5['first_cycle_steps']})")
    check(s5["warmup_steps"] != s4["warmup_steps"]
          or s5["first_cycle_steps"] != s4["first_cycle_steps"],
          "'keep' and 'rescale' produce genuinely different schedules")


if __name__ == "__main__":
    _spec = os.environ.get("OO_E2E_CHILD")
    if _spec:
        sys.exit(run_child(json.loads(Path(_spec).read_text(encoding="utf-8"))))
    sys.exit(main())
