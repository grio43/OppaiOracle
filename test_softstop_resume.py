"""Regression tests for the soft-stop / resume / epoch+step-tracking logic.

Run directly:  python test_softstop_resume.py

Covers, without launching training:
  * ResumableSampler offset semantics (consumed once, no leak, incl. GeneratorExit)
  * len() agreement with a real DataLoader, and the deliberate len-vs-offset-pass
    disagreement that replaced the old __len__ override
  * the latch-then-act soft-stop rule, using train_direct's own SENTINEL_CHECK_INTERVAL
  * a faithful state-machine simulation of the soft-stop / accumulation-carry
    interaction across epoch boundaries, contrasted against the pre-fix form
  * the should_validate predicate and the zero-batch / burn-in guards
  * the signal-handler escalation contract, asserted against the real AST
  * TrainingState round-trip and legacy-checkpoint compatibility
"""
import ast
import math
import re
import sys
import threading
from pathlib import Path

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))

import torch  # noqa: F401  (DistributedSampler needs torch.distributed importable)

from dataset_loader import ResumableSampler, DataLoader as GuardedDataLoader

_TRAIN_SRC = (REPO / "train_direct.py").read_text(encoding="utf-8")
_TRAIN_AST = ast.parse(_TRAIN_SRC)


class _DS:
    def __init__(self, n):
        self.n = n

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        return i


def _sentinel_interval():
    """Read SENTINEL_CHECK_INTERVAL from train_direct.py without importing it.

    Importing train_direct pulls in the whole training stack. Fails loudly (rather
    than silently defaulting) if the constant stops being an int literal.
    """
    m = re.search(r"^\s*SENTINEL_CHECK_INTERVAL\s*=\s*(\d+)", _TRAIN_SRC, re.M)
    assert m, "SENTINEL_CHECK_INTERVAL is no longer an int literal in train_direct.py"
    return int(m.group(1))


# =========================================================================== sampler


def test_sampler_offset():
    ds = _DS(100)
    s = ResumableSampler(ds, num_replicas=1, rank=0, shuffle=True, seed=0, drop_last=False)
    s.set_epoch(3)
    full = list(s)
    assert len(full) == 100, len(full)

    s.set_epoch(3)
    s.set_start_index(60)
    assert list(s) == full[60:], "offset pass must equal the tail of the full pass"

    s.set_epoch(3)
    assert list(s) == full, "offset leaked into the following epoch"
    print("PASS: offset applied once, no leak (exhausted iterator)")


def test_sampler_offset_abandoned():
    """The regression: abandoning the iterator skipped the trailing reset."""
    ds = _DS(100)
    s = ResumableSampler(ds, num_replicas=1, rank=0, shuffle=True, seed=0, drop_last=False)
    s.set_epoch(5)
    full = list(s)

    s.set_epoch(5)
    s.set_start_index(60)
    it = iter(s)
    next(it)
    next(it)
    del it  # abandon -> GeneratorExit, trailing statements never run

    s.set_epoch(5)
    assert list(s) == full, "offset leaked after an abandoned iterator"
    print("PASS: offset does not leak after an abandoned iterator")


def test_len_semantics_against_real_dataloader():
    """len() must equal a FULL pass and must deliberately disagree with an offset pass.

    Asserting both directions locks in the post-fix contract: the __len__ override was
    removed because it was only valid between set_start_index() and the first next().
    Re-introducing any length override would fail one side or the other.
    """
    offset = 64
    for drop_last in (False, True):
        for num_workers in (0, 2):
            ds = _DS(100)
            s = ResumableSampler(ds, num_replicas=1, rank=0, shuffle=True,
                                 seed=0, drop_last=drop_last)
            s.set_epoch(0)
            n_indices = len(list(s))
            # Uses the repo's guarded DataLoader wrapper so the num_workers==0
            # kwarg-stripping path is exercised too.
            loader = GuardedDataLoader(
                ds, batch_size=8, sampler=s, drop_last=drop_last,
                num_workers=num_workers, persistent_workers=True, prefetch_factor=2,
            )
            declared = len(loader)
            assert declared == sum(1 for _ in loader), (
                f"drop_last={drop_last} nw={num_workers}: len={declared} != full pass"
            )

            s.set_epoch(0)
            s.set_start_index(offset)
            before = len(loader)
            remaining = n_indices - offset
            expected = remaining // 8 if drop_last else math.ceil(remaining / 8)
            yielded = sum(1 for _ in loader)
            assert yielded == expected, (
                f"drop_last={drop_last} nw={num_workers}: offset pass yielded {yielded}, "
                f"expected {expected}"
            )
            assert before == declared, "len() must not track the pending offset"
            assert len(loader) == declared, "len() must be stable across iteration"
            del loader
    print("PASS: len() == full pass, and deliberately != offset pass (real DataLoader)")


def test_get_state_reports_pending_offset_only():
    ds = _DS(100)
    s = ResumableSampler(ds, num_replicas=1, rank=0, shuffle=True, seed=0, drop_last=False)
    s.set_epoch(1)
    s.set_start_index(40)
    assert s.get_state()['start_index'] == 40, "pending offset must be visible before iteration"
    it = iter(s)
    next(it)
    st = s.get_state()
    assert st['start_index'] == 0, f"consumed offset must read 0, got {st}"
    assert st['total_size'] == 100, "total_size is the field the trainer guards on"
    del it
    print("PASS: get_state() reports the PENDING offset, 0 once consumed")


# ====================================================== resume with a CHANGED batch size


def _resume_skip(sample_in_epoch, batch_in_epoch, cur_bs, ckpt_bs=None, buggy=False):
    """Mirror train_direct.py's mid-epoch resume block.

    Returns (total samples skipped before the first trained batch, start_step,
    sample_pos_correction).

    buggy=True reproduces the pre-fix gate on the slow fallback path
    (`start_step == 0` instead of `not sampler_offset_applied`).
    """
    start_step = 0
    sampler_offset_applied = False
    sample_pos_correction = 0
    skipped = 0

    if batch_in_epoch > 0 or sample_in_epoch > 0:
        # ResumableSampler is always installed by create_dataloaders, so the
        # set_start_index branch is the one that runs.
        if sample_in_epoch > 0:
            sample_offset = sample_in_epoch
        else:
            sample_offset = batch_in_epoch * (ckpt_bs or cur_bs)
        start_step = sample_offset // cur_bs
        sampler_offset_applied = True
        sample_pos_correction = sample_offset - start_step * cur_bs
        skipped += sample_offset

    take_fallback = (batch_in_epoch > 0 and
                     (start_step == 0 if buggy else not sampler_offset_applied))
    if take_fallback:
        skipped += batch_in_epoch * cur_bs
        start_step = batch_in_epoch

    return skipped, start_step, sample_pos_correction


def test_resume_offset_is_not_double_applied_after_a_batch_size_change():
    """The fallback must not skip batches on top of an applied sampler offset.

    Pre-fix the fallback was gated on `start_step == 0`, which is also true when the
    recorded sample offset is SMALLER than the current batch size - i.e. exactly when
    batch_size was increased between the stop and the resume. The offset had already
    been applied, so the fallback consumed resume_batch_idx more batches on top of it
    and silently dropped them from the replayed epoch.
    """
    victims = []
    for old_bs in (8, 32, 48, 96):
        for new_bs in (8, 32, 48, 96, 128, 512):
            for stop_step in (0, 1, 5, 9, 40, 300):
                bie = stop_step + 1
                sie = bie * old_bs
                fixed, start_step, corr = _resume_skip(sie, bie, new_bs)
                assert fixed == sie, (
                    f"post-fix skipped {fixed} samples, expected exactly {sie} "
                    f"(old_bs={old_bs} new_bs={new_bs} stop_step={stop_step})"
                )
                # The recorded position must be exactly reconstructible.
                assert start_step * new_bs + corr == sie, (
                    f"position accounting lost {sie - (start_step * new_bs + corr)} samples"
                )
                assert 0 <= corr < new_bs, corr

                old, _, _ = _resume_skip(sie, bie, new_bs, buggy=True)
                if old != sie:
                    victims.append((old_bs, new_bs, stop_step, old - sie))

    assert victims, "grid never reproduced the pre-fix double-skip - test is vacuous"
    worst = max(v[3] for v in victims)
    # Only reachable when the batch size GREW past the recorded offset.
    assert all(new > old for old, new, _, _ in victims), victims
    print(f"PASS: sampler offset applied exactly once across batch-size changes "
          f"(pre-fix form double-skipped in {len(victims)} shapes, up to {worst} samples)")


def test_resume_offset_unchanged_batch_size_is_untouched():
    """At constant batch size the fallback was never reachable; keep it that way."""
    for bs in (8, 48, 96, 512):
        for stop_step in (0, 1, 7, 12499):
            bie = stop_step + 1
            sie = bie * bs
            for buggy in (True, False):
                skipped, start_step, corr = _resume_skip(sie, bie, bs, buggy=buggy)
                assert skipped == sie and start_step == bie and corr == 0, (
                    f"bs={bs} stop_step={stop_step} buggy={buggy}: "
                    f"{skipped}/{start_step}/{corr}"
                )
    print("PASS: constant-batch-size resume is byte-identical to the pre-fix path")


def test_sample_in_epoch_records_the_true_position():
    """Every mid-epoch save must record the exact sample position, not a floored one.

    Without the correction term, `(step + 1) * batch_size` is
    `sample_offset % batch_size` samples behind the truth for the whole replayed
    epoch, so a second stop/resume cycle replays that many samples.
    """
    old_bs, new_bs = 48, 128
    bie = 10
    sie = bie * old_bs                      # 480, not a multiple of 128
    _, start_step, corr = _resume_skip(sie, bie, new_bs)
    assert corr != 0, "pick a shape where the offset is genuinely misaligned"

    for batches_done in (1, 2, 37):
        step = start_step + batches_done - 1
        true_pos = sie + batches_done * new_bs
        assert (step + 1) * new_bs + corr == true_pos, "corrected form must be exact"
        assert (step + 1) * new_bs != true_pos, "uncorrected form should be wrong here"

    # And the three mid-epoch save sites must all carry the term.
    sites = [n for n in ast.walk(_TRAIN_AST)
             if isinstance(n, ast.Assign)
             and any(isinstance(t, ast.Attribute) and t.attr == 'sample_in_epoch'
                     for t in n.targets)]
    mid = [n for n in sites
           if not (isinstance(n.value, ast.Constant) and n.value.value == 0)]
    assert len(mid) == 3, f"expected 3 mid-epoch sample_in_epoch writes, found {len(mid)}"
    for n in mid:
        assert any(isinstance(s, ast.Name) and s.id == 'sample_pos_correction'
                   for s in ast.walk(n.value)), (
            f"sample_in_epoch write at line {n.lineno} drops sample_pos_correction"
        )
    print("PASS: mid-epoch saves record the exact sample position (all 3 sites)")


def test_fallback_is_gated_on_the_applied_flag_not_start_step():
    """Source-level lock so the `start_step == 0` proxy cannot creep back in."""
    assert "sampler_offset_applied = True" in _TRAIN_SRC
    assert "and not sampler_offset_applied:" in _TRAIN_SRC, (
        "the slow fallback must be gated on sampler_offset_applied"
    )
    assert "resume_batch_idx > 0 and start_step == 0" not in _TRAIN_SRC, (
        "the pre-fix `start_step == 0` gate is back - it double-skips when batch_size grew"
    )
    print("PASS: fallback gate is the applied-flag, not the floored start_step")


# ================================================ LR schedule across a batch-size change


def _sched(peak, warmup, cycle, min_lr=1e-6, gamma=0.9):
    import torch.nn as nn
    from training_utils import CosineAnnealingWarmupRestarts
    m = nn.Linear(4, 4)
    opt = torch.optim.SGD(m.parameters(), lr=peak)
    s = CosineAnnealingWarmupRestarts(
        opt, first_cycle_steps=cycle, cycle_mult=1.0, max_lr=peak,
        min_lr=min_lr, warmup_steps=warmup, gamma=gamma,
    )
    return opt, s


def test_retarget_rescales_lr_and_completes_the_anneal():
    """A resume at a different effective batch must land on min_lr, at the new peak.

    Keeping the checkpoint's geometry (the pre-fix behaviour) does neither: the peak
    stays the old one, and halving the effective batch doubles the updates so the run
    overruns cur_cycle_steps and fires an unintended SGDR restart - which the config
    documents as impossible at num_cycles=1.
    """
    UPE_OLD, EPOCHS, WARM_E = 12499, 15, 2
    OLD_PEAK, NEW_PEAK = 1.30e-5, 0.919e-5      # sqrt-scaled for a halved batch

    _, s_old = _sched(OLD_PEAK, WARM_E * UPE_OLD, EPOCHS * UPE_OLD)
    for _ in range(5 * UPE_OLD):
        s_old.step()
    saved = s_old.state_dict()
    frac_old = s_old.step_in_cycle / s_old.cur_cycle_steps

    UPE_NEW = 2 * UPE_OLD
    warm_new, cycle_new = WARM_E * UPE_NEW, EPOCHS * UPE_NEW

    # keep-mode: the defect this exists to prevent
    opt_k, s_keep = _sched(NEW_PEAK, warm_new, cycle_new)
    s_keep.load_state_dict(dict(saved))
    assert s_keep.base_max_lr == OLD_PEAK, "load_state_dict should restore the old peak"
    cycles_before = s_keep.cycle
    for _ in range((EPOCHS - 5) * UPE_NEW):
        s_keep.step()
    assert s_keep.cycle > cycles_before, (
        "expected keep-mode to overrun the restored cycle and warm-restart"
    )

    # rescale-mode
    opt_r, s_new = _sched(NEW_PEAK, warm_new, cycle_new)
    s_new.load_state_dict(dict(saved))
    s_new.retarget(max_lr=NEW_PEAK, warmup_steps=warm_new, first_cycle_steps=cycle_new)

    assert s_new.base_max_lr == NEW_PEAK, "peak must follow the current config"
    assert s_new.warmup_steps == warm_new and s_new.cur_cycle_steps == cycle_new
    frac_new = s_new.step_in_cycle / s_new.cur_cycle_steps
    assert abs(frac_new - frac_old) < 1e-3, (
        f"progress not preserved: {frac_old:.6f} -> {frac_new:.6f}"
    )
    # The live LR must be updated immediately - the next optimizer step precedes the
    # next scheduler.step(), so a stale restored LR would be used for one update.
    assert abs(opt_r.param_groups[0]['lr'] - s_new.get_lr()[0]) < 1e-15
    assert opt_r.param_groups[0]['lr'] < OLD_PEAK

    # Walk to one update short of the cycle end - i.e. the last update this run will
    # actually perform. A restart before that point is the defect keep-mode exhibits.
    cycles_before = s_new.cycle
    remaining = cycle_new - s_new.step_in_cycle - 1
    assert remaining > 0
    for _ in range(remaining):
        s_new.step()
    assert s_new.cycle == cycles_before, "retargeted schedule restarted before the end"
    assert s_new.step_in_cycle == cycle_new - 1
    final = opt_r.param_groups[0]['lr']
    assert abs(final - s_new.min_lr) < 0.05 * NEW_PEAK, (
        f"anneal did not reach min_lr: ended at {final:.3e}"
    )
    print(f"PASS: retarget rescales the peak ({OLD_PEAK:.2e} -> {NEW_PEAK:.2e}), "
          f"preserves {frac_new:.1%} progress and lands on min_lr; "
          f"keep-mode warm-restarts instead")


def test_effective_batch_size_is_never_shadowed():
    """`effective_batch_size` must keep ONE meaning for the whole of train().

    The legacy-checkpoint resume branch used to reassign it to a plain batch_size
    (48 instead of 48*9=432). Every reader happened to run earlier, so nothing broke -
    but the effective-batch guard reads it, and any future reader inside the epoch loop
    would silently get samples-per-batch where samples-per-update was meant.
    """
    fns = [n for n in ast.walk(_TRAIN_AST)
           if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
    host = next(f for f in fns if f.name == 'train_with_orientation_tracking')
    stores = [n.lineno for n in ast.walk(host)
              if isinstance(n, ast.Name) and n.id == 'effective_batch_size'
              and isinstance(n.ctx, ast.Store)]
    assert len(stores) == 1, (
        f"effective_batch_size is assigned {len(stores)} times (lines {stores}); it must "
        f"mean batch_size * gradient_accumulation_steps everywhere in train()"
    )
    print("PASS: effective_batch_size has a single binding (no mid-run shadowing)")


def test_retarget_syncs_initial_lr_into_the_optimizer():
    """initial_lr rides into every later checkpoint via optimizer.state_dict()."""
    import torch.nn as nn
    from training_utils import CosineAnnealingWarmupRestarts
    m = nn.Linear(4, 4)
    opt = torch.optim.SGD(m.parameters(), lr=1e-5)
    s = CosineAnnealingWarmupRestarts(opt, first_cycle_steps=1000, cycle_mult=1.0,
                                      max_lr=1e-5, min_lr=1e-7, warmup_steps=100, gamma=0.9)
    assert 'initial_lr' in opt.param_groups[0], "torch stopped setting initial_lr"
    s.retarget(max_lr=2e-5, warmup_steps=200, first_cycle_steps=2000)
    for group, base in zip(opt.param_groups, s.base_lrs):
        assert group['initial_lr'] == base, (
            f"initial_lr={group['initial_lr']} contradicts base_lr={base} after retarget"
        )
    assert opt.param_groups[0]['initial_lr'] == 2e-5
    print("PASS: retarget keeps optimizer initial_lr consistent with base_lrs")


def test_retarget_preserves_per_group_lr_ratios():
    """Layer-wise decay / separate head LR must survive the retarget."""
    import torch.nn as nn
    from training_utils import CosineAnnealingWarmupRestarts
    m = nn.Linear(4, 4)
    opt = torch.optim.SGD([
        {'params': [m.weight], 'lr': 1e-5},
        {'params': [m.bias], 'lr': 2e-5},
    ])
    s = CosineAnnealingWarmupRestarts(opt, first_cycle_steps=1000, cycle_mult=1.0,
                                      max_lr=2e-5, min_lr=1e-7, warmup_steps=100, gamma=0.9)
    before = list(s.base_lrs)
    ratio_before = before[0] / before[1]
    s.retarget(max_lr=4e-5, warmup_steps=200, first_cycle_steps=2000)
    after = list(s.base_lrs)
    assert abs(after[0] / after[1] - ratio_before) < 1e-12, (after, before)
    assert abs(after[1] / before[1] - 2.0) < 1e-12, "peak did not double with max_lr"
    print("PASS: retarget preserves per-param-group LR ratios")


def _exec_lr_schedule_guard(*, ckpt_bs=48, ckpt_accum=9, cur_bs=48, cur_accum=9,
                            policy='error',
                            cfg_peak=0.919e-5, cfg_warmup=49996, cfg_cycle=374970,
                            cfg_min_lr=1e-6, cfg_gamma=0.9,
                            sched_peak=None, sched_warmup=None, sched_cycle=None,
                            sched_min_lr=None, sched_gamma=None,
                            sched_peak_missing=False,
                            scheduler_restored=True, ckpt_batch_recorded=True):
    """Execute train_direct's REAL LR-schedule guard against a stub namespace.

    The guard lives inline in a 2800-line function, so it is lifted out by AST and run
    directly. That keeps this test honest: it exercises the shipped branch structure,
    not a paraphrase of it.

    The cfg_* values are what the CURRENT config computes; the sched_* values are what
    load_state_dict() put on the live scheduler (they default to the cfg_* values, i.e.
    no drift). Splitting the two is the whole point of the broadened guard: a dataset
    that grew or shrank moves cfg_warmup / cfg_cycle while every batch number stays put.
    """
    guard = [n for n in ast.walk(_TRAIN_AST) if isinstance(n, ast.If)
             and any(isinstance(x, ast.Name) and x.id == 'ckpt_scheduler_restored'
                     and isinstance(x.ctx, ast.Load) for x in ast.walk(n.test))]
    assert len(guard) == 1, f"expected exactly one LR-schedule guard, found {len(guard)}"
    mod = ast.Module(body=[guard[0]], type_ignores=[])
    ast.fix_missing_locations(mod)

    logs = []

    class _Log:
        def warning(self, msg, *a):
            logs.append(msg % a if a else msg)

    class _NS:
        def __init__(self, **kw):
            self.__dict__.update(kw)

    _, sched = _sched(
        cfg_peak if sched_peak is None else sched_peak,
        cfg_warmup if sched_warmup is None else sched_warmup,
        cfg_cycle if sched_cycle is None else sched_cycle,
        min_lr=cfg_min_lr if sched_min_lr is None else sched_min_lr,
        gamma=cfg_gamma if sched_gamma is None else sched_gamma,
    )
    if sched_peak_missing:
        # A checkpoint predating base_max_lr leaves the attribute unobservable. The
        # guard must skip that dimension rather than read None as a mismatch.
        sched.base_max_lr = None
    ns = {
        'math': math,
        'ckpt_scheduler_restored': scheduler_restored,
        'ckpt_effective_batch': (ckpt_bs, ckpt_accum) if ckpt_batch_recorded else None,
        'effective_batch_size': cur_bs * cur_accum,
        'accum': cur_accum,
        'effective_learning_rate': cfg_peak,
        'warmup_steps': cfg_warmup,
        'first_cycle_steps': cfg_cycle,
        'updates_per_epoch': 24998,
        'num_epochs': 15,
        'warmup_epochs': 2,
        'num_cycles': 1,
        'scheduler': sched,
        'logger': _Log(),
        'config': _NS(
            data=_NS(batch_size=cur_bs),
            # lr_end / cycle_decay must be present and must match what _sched built,
            # or every caller drifts on those axes. A stub that omits them would read
            # the guard's getattr defaults (1e-6 / 1.0) instead of the real config, and
            # a stub wrong in the same way as a bug hides that bug.
            training=_NS(on_lr_schedule_change=policy, eval_steps=11538,
                         lr_end=cfg_min_lr, cycle_decay=cfg_gamma),
        ),
    }
    exec(compile(mod, '<guard>', 'exec'), ns)
    return logs, sched


def test_guard_catches_min_lr_and_gamma_drift():
    """lr_end and cycle_decay are restored by load_state_dict like every other field.

    They were skipped while the guard was scoped to batch-size changes (neither depends
    on batch size). Once it became a general LR-schedule guard that was a hole: a run
    that lowered training.lr_end resumed onto the OLD floor, annealed to it, and neither
    the guard nor retarget() said a word.
    """
    # min_lr drift alone must fire.
    try:
        _exec_lr_schedule_guard(cfg_min_lr=1e-7, sched_min_lr=1e-6, policy='error')
    except RuntimeError as e:
        assert 'min_lr' in str(e) and 'lr_end' in str(e), str(e)
    else:
        raise AssertionError("a changed lr_end slipped past the guard")

    # gamma drift alone must fire.
    try:
        _exec_lr_schedule_guard(cfg_gamma=0.5, sched_gamma=0.9, policy='error')
    except RuntimeError as e:
        assert 'gamma' in str(e) and 'cycle_decay' in str(e), str(e)
    else:
        raise AssertionError("a changed cycle_decay slipped past the guard")

    # rescale must actually apply both, not just report them.
    logs, s = _exec_lr_schedule_guard(cfg_min_lr=1e-7, sched_min_lr=1e-6,
                                      cfg_gamma=0.5, sched_gamma=0.9, policy='rescale')
    assert logs, "rescale must announce what it did"
    assert s.min_lr == 1e-7, f"retarget left min_lr at {s.min_lr}"
    assert s.gamma == 0.5, f"retarget left gamma at {s.gamma}"

    # keep must leave both alone.
    logs, s = _exec_lr_schedule_guard(cfg_min_lr=1e-7, sched_min_lr=1e-6, policy='keep')
    assert logs and 'keep' in logs[0], logs
    assert s.min_lr == 1e-6, "keep must not modify the restored schedule"

    # And matching values on both axes stay silent.
    logs, _ = _exec_lr_schedule_guard(cfg_min_lr=1e-7, sched_min_lr=1e-7,
                                      cfg_gamma=0.5, sched_gamma=0.5, policy='error')
    assert not logs, logs
    print("PASS: guard catches min_lr / gamma drift; rescale applies both, keep does not")


def test_min_lr_ordering_inside_retarget():
    """min_lr/gamma must be assigned BEFORE max_lr is recomputed from them.

    max_lr = max(base_max_lr * gamma**cycle, min_lr). Setting either after that line
    would leave max_lr computed from the stale value - invisible at cycle==0, wrong the
    moment a restart has happened.
    """
    import torch.nn as nn
    from training_utils import CosineAnnealingWarmupRestarts
    m = nn.Linear(4, 4)
    opt = torch.optim.SGD(m.parameters(), lr=1e-5)
    s = CosineAnnealingWarmupRestarts(opt, first_cycle_steps=1000, cycle_mult=1.0,
                                      max_lr=1e-5, min_lr=1e-7, warmup_steps=100, gamma=0.9)
    s.cycle = 2  # pretend two restarts already happened
    s.retarget(max_lr=2e-5, warmup_steps=200, first_cycle_steps=2000, gamma=0.5)
    assert s.gamma == 0.5
    assert abs(s.max_lr - 2e-5 * 0.5 ** 2) < 1e-18, (
        f"max_lr={s.max_lr} was computed from the stale gamma (expected {2e-5 * 0.25})"
    )
    # min_lr acting as the floor must also use the NEW value.
    s2 = CosineAnnealingWarmupRestarts(
        torch.optim.SGD(nn.Linear(2, 2).parameters(), lr=1e-5),
        first_cycle_steps=1000, cycle_mult=1.0, max_lr=1e-5,
        min_lr=1e-7, warmup_steps=100, gamma=0.5)
    s2.cycle = 40  # gamma**40 underflows the peak far below any sane floor
    s2.retarget(max_lr=2e-5, warmup_steps=200, first_cycle_steps=2000, min_lr=3e-6)
    assert s2.max_lr == 3e-6, f"floor not applied from the new min_lr: {s2.max_lr}"
    print("PASS: retarget assigns min_lr/gamma before recomputing max_lr")


def test_guard_arming_line_is_branch_correct():
    """Lock the arming of `ckpt_scheduler_restored` structurally.

    The guard-execution tests inject this flag into the namespace, so the line that
    SETS it is never exercised by them. Hoisting it out of the phase-transition
    else-branch, or hardcoding it True, would leave every one of those tests green
    while arming the guard on a phase transition - where optimizer/scheduler are passed
    as None to load_checkpoint and deliberately start fresh, so there is no restored
    geometry to compare against and every dimension would read as drift.
    """
    fn = next(n for n in ast.walk(_TRAIN_AST)
              if isinstance(n, ast.FunctionDef)
              and n.name == 'train_with_orientation_tracking')
    stores = [n for n in ast.walk(fn) if isinstance(n, ast.Assign)
              and any(isinstance(t, ast.Name) and t.id == 'ckpt_scheduler_restored'
                      for t in n.targets)]
    assert len(stores) == 2, f"expected exactly 2 bindings, found {len(stores)}"

    init = [a for a in stores if isinstance(a.value, ast.Constant) and a.value.value is False]
    arm = [a for a in stores if a not in init]
    assert len(init) == 1 and len(arm) == 1, "expected one `= False` init and one arming"

    # The arming must test the checkpoint for scheduler state, not be hardcoded.
    assert isinstance(arm[0].value, ast.Compare) and any(
        isinstance(o, ast.In) for o in arm[0].value.ops
    ), "arming must be `'scheduler_state_dict' in ckpt`, not a constant"

    # ...and it must sit in the ELSE branch of `if phase_transition:`.
    holder = [n for n in ast.walk(fn) if isinstance(n, ast.If)
              and any(isinstance(x, ast.Name) and x.id == 'phase_transition'
                      for x in ast.walk(n.test))
              and any(arm[0] is s for b in n.orelse for s in ast.walk(b))]
    assert holder, (
        "ckpt_scheduler_restored is armed outside the `if phase_transition: ... else:` "
        "branch - a phase transition would arm the guard against a fresh scheduler"
    )
    print("PASS: guard arming is inside the non-phase-transition branch (AST-locked)")


def test_effective_batch_guard_branches():
    """The shipped guard: no-op when unchanged, raises by default, warns on keep,
    retargets on rescale."""
    NEW_PEAK = 0.919e-5

    # nothing drifted -> completely silent, scheduler untouched
    for policy in ('error', 'keep', 'rescale'):
        logs, s = _exec_lr_schedule_guard(policy=policy)
        assert not logs, f"policy={policy} spoke up for an unchanged schedule: {logs}"

    # changed + default policy -> refuse, with actionable text
    try:
        _exec_lr_schedule_guard(ckpt_bs=48, ckpt_accum=9, cur_bs=96, cur_accum=9,
                                policy='error')
    except RuntimeError as e:
        msg = str(e)
        assert '48x9=432' in msg and '96x9=864' in msg, msg
        assert 'effective batch size' in msg, f"drifted dimension not named:\n{msg}"
        for hint in ("'rescale'", "'keep'", "resume_from: 'none'", 'eval_steps',
                     'data.batch_size=48'):
            assert hint in msg, f"error message omits {hint}:\n{msg}"
    else:
        raise AssertionError("changed effective batch did not raise under the default policy")

    # an unrecognised policy string must fail closed, not fall through silently
    try:
        _exec_lr_schedule_guard(ckpt_bs=48, ckpt_accum=9, cur_bs=96, cur_accum=9,
                                policy='yolo')
    except RuntimeError:
        pass
    else:
        raise AssertionError("unknown policy silently continued")

    # keep -> warns, leaves the schedule alone
    logs, s = _exec_lr_schedule_guard(ckpt_bs=48, ckpt_accum=9, cur_bs=96,
                                      cur_accum=9, policy='keep')
    assert logs and 'keep' in logs[0], logs
    assert s.base_max_lr == NEW_PEAK, "keep must not touch the scheduler"

    # rescale -> warns AND retargets
    logs, s = _exec_lr_schedule_guard(ckpt_bs=48, ckpt_accum=9, cur_bs=96,
                                      cur_accum=9, policy='rescale',
                                      cfg_peak=NEW_PEAK, cfg_warmup=1234,
                                      cfg_cycle=56789)
    assert logs and 'Retarget' in logs[0], logs
    assert s.warmup_steps == 1234 and s.cur_cycle_steps == 56789, (
        f"rescale did not apply the new geometry: {s.warmup_steps}/{s.cur_cycle_steps}"
    )
    assert s.base_max_lr == NEW_PEAK
    print("PASS: shipped LR-schedule guard is silent/raises/warns/retargets correctly")


def test_lr_schedule_guard_fires_on_dataset_size_drift():
    """A dataset that grew or shrank must trip the guard at an IDENTICAL batch size.

    This is the axis the original batch-only trigger was blind to: warmup_steps and
    first_cycle_steps come from updates_per_epoch, which comes from len(train_loader),
    so adding or removing training images silently invalidates the restored schedule
    while every batch number the old guard looked at stays exactly the same.
    """
    # +20% images: same 48x9 batch, but warmup and cycle both move.
    try:
        _exec_lr_schedule_guard(policy='error',
                                cfg_warmup=59995, cfg_cycle=449964,
                                sched_warmup=49996, sched_cycle=374970)
    except RuntimeError as e:
        msg = str(e)
        assert 'warmup_steps: current 59995 vs checkpoint 49996' in msg, msg
        assert 'first_cycle_steps: current 449964 vs checkpoint 374970' in msg, msg
        assert 'effective batch size' not in msg, (
            f"named a dimension that did not drift:\n{msg}"
        )
        assert 'updates_per_epoch' in msg, f"no restore hint for the geometry axis:\n{msg}"
        assert 'data.batch_size=' not in msg, (
            f"told the operator to restore a batch size that never moved:\n{msg}"
        )
    else:
        raise AssertionError("dataset-size drift did not raise under the default policy")

    # warmup alone is enough; the cycle can be unchanged (e.g. warmup_epochs edited)
    try:
        _exec_lr_schedule_guard(policy='error', cfg_warmup=49996, sched_warmup=24998)
    except RuntimeError as e:
        assert 'warmup_steps' in str(e) and 'first_cycle_steps' not in str(e), str(e)
    else:
        raise AssertionError("warmup-only drift did not raise")

    # and 'rescale' must fix the geometry axis too, not just the batch axis
    logs, s = _exec_lr_schedule_guard(policy='rescale',
                                      cfg_warmup=59995, cfg_cycle=449964,
                                      sched_warmup=49996, sched_cycle=374970)
    assert logs and 'Retarget' in logs[0], logs
    assert s.warmup_steps == 59995 and s.cur_cycle_steps == 449964, (
        f"rescale left the stale geometry: {s.warmup_steps}/{s.cur_cycle_steps}"
    )
    print("PASS: LR-schedule guard fires on pure dataset-size drift and rescales it")


def test_lr_schedule_guard_peak_lr_axis():
    """base_max_lr drift fires; an absent base_max_lr must NOT be read as drift."""
    # A hand-edited training.learning_rate with everything else identical.
    try:
        _exec_lr_schedule_guard(policy='error', cfg_peak=1.30e-5, sched_peak=0.919e-5)
    except RuntimeError as e:
        msg = str(e)
        assert 'base_max_lr' in msg, msg
        assert '1.300000e-05' in msg and '9.190000e-06' in msg, msg
        assert 'training.learning_rate' in msg, f"no restore hint for the LR axis:\n{msg}"
    else:
        raise AssertionError("peak-LR drift did not raise under the default policy")

    # Float noise must not be mistaken for drift: the comparison is math.isclose, and a
    # peak that has been through scale_learning_rate() + a save/load round-trip will not
    # be bit-identical to the freshly computed one.
    logs, _ = _exec_lr_schedule_guard(policy='error', cfg_peak=0.919e-5,
                                      sched_peak=0.919e-5 * (1 + 1e-13))
    assert not logs, f"bit-level float noise was reported as drift: {logs}"

    # A legacy checkpoint with no base_max_lr: unobservable is not the same as drifted.
    logs, _ = _exec_lr_schedule_guard(policy='error', sched_peak_missing=True)
    assert not logs, f"absent base_max_lr was treated as drift: {logs}"

    # ...but a missing peak must not mask a real drift on another dimension.
    try:
        _exec_lr_schedule_guard(policy='error', sched_peak_missing=True,
                                cfg_warmup=59995, sched_warmup=49996)
    except RuntimeError as e:
        assert 'warmup_steps' in str(e) and 'base_max_lr' not in str(e), str(e)
    else:
        raise AssertionError("absent base_max_lr suppressed a real warmup drift")
    print("PASS: LR-schedule guard checks base_max_lr with a tolerance and skips it when absent")


def test_lr_schedule_guard_is_disarmed_without_restored_state():
    """No restored scheduler state -> no comparison, however far the config has moved.

    On a phase transition the scheduler is passed to load_checkpoint() as None and keeps
    the geometry THIS config just built, so any 'drift' the guard measured would be
    against itself. Same for a checkpoint saved with save_scheduler=False.
    """
    logs, s = _exec_lr_schedule_guard(scheduler_restored=False, policy='error',
                                      ckpt_bs=48, ckpt_accum=9, cur_bs=96, cur_accum=9,
                                      cfg_peak=1.30e-5, sched_peak=0.919e-5,
                                      cfg_warmup=59995, sched_warmup=49996)
    assert not logs, f"guard fired with no restored scheduler state: {logs}"
    assert s.warmup_steps == 49996, "guard touched the scheduler while disarmed"

    # A checkpoint with restored scheduler state but no embedded config still gets its
    # scheduler-derived dimensions checked - the batch dimension is simply skipped.
    try:
        _exec_lr_schedule_guard(ckpt_batch_recorded=False, policy='error',
                                cfg_warmup=59995, sched_warmup=49996)
    except RuntimeError as e:
        assert 'warmup_steps' in str(e) and 'effective batch size' not in str(e), str(e)
    else:
        raise AssertionError("geometry drift was skipped just because the config was absent")
    print("PASS: LR-schedule guard is disarmed on phase transitions / unsaved scheduler state")


def test_effective_batch_policy_is_wired():
    """All three policies must exist, and 'error' must be the default."""
    from Configuration_System import TrainingConfig
    assert TrainingConfig().on_lr_schedule_change == "error", (
        "the default must refuse, not silently mis-train"
    )
    assert not hasattr(TrainingConfig(), "on_effective_batch_change"), (
        "the old batch-only key name must be gone, not left as a silent no-op alias"
    )
    for policy in ("'keep'", "'rescale'"):
        assert f"_policy == {policy}" in _TRAIN_SRC, f"policy {policy} not handled"
    assert "scheduler.retarget(" in _TRAIN_SRC, "rescale path never calls retarget()"
    assert "'base_max_lr'" in (REPO / "training_utils.py").read_text(encoding="utf-8"), (
        "base_max_lr must be checkpointed so the resume-time diff check can compare it"
    )
    # The key must be spelled identically in the YAML the run actually loads; a rename
    # that misses the config file leaves the default silently in force.
    _yaml = (REPO / "configs" / "unified_config.yaml").read_text(encoding="utf-8")
    assert "on_lr_schedule_change:" in _yaml, "renamed key missing from unified_config.yaml"
    assert "on_effective_batch_change" not in _yaml, "stale key name left in unified_config.yaml"
    print("PASS: LR-schedule-change policy is wired, default is 'error'")


# ============================================================ soft-stop latch (arithmetic)


def _latch_sim(accum, interval, steps, latched, skip_every=0):
    """Step at which a mid-epoch soft-stop checkpoint would be written, or None.

    latched=True  -> current behaviour (sentinel sets a sticky event, polled every step)
    latched=False -> pre-fix behaviour (throttled expression tested directly)
    skip_every>0  -> every Nth microbatch takes a `continue` path (failed load / NaN
                     loss / non-finite grad), zeroing accum_count and skipping the check.
    Models the arithmetic only; the stall bail-out is covered separately.
    """
    event = False
    accum_count = 0
    for step in range(steps):
        # The sentinel poll sits at the TOP of the microbatch body, above the
        # `continue` paths, so a failure pattern covering the poll steps cannot hide
        # the request. Only the latch is hoisted; the save stays at the bottom.
        if latched and step % interval == 0:
            event = True
        if skip_every and step % skip_every == 0:
            accum_count = 0
            continue
        accum_count += 1
        if accum_count >= accum:
            accum_count = 0
        stop_requested = event if latched else (step % interval == 0)
        if stop_requested and accum_count == 0:
            return step
    return None


def test_stop_reachability():
    interval = _sentinel_interval()
    broken_old = []
    for accum in range(1, 25):
        old = _latch_sim(accum, interval, 20000, latched=False)
        new = _latch_sim(accum, interval, 20000, latched=True)
        assert new is not None, f"latched behaviour failed to stop at accum={accum}"
        assert new <= accum, f"latched stop latency at accum={accum} was {new}, expected <= {accum}"
        if old is None:
            broken_old.append(accum)
    print(f"PASS: latched stop reachable for every accum 1..24 (interval={interval}), "
          f"latency <= accum")
    print(f"      pre-fix form could NEVER stop mid-epoch for accum in {broken_old}")
    assert 8 in broken_old and 10 in broken_old, broken_old
    assert 9 not in broken_old, "accum=9 (current config) happened to work pre-fix"


def test_stop_survives_skipped_batches():
    interval = _sentinel_interval()
    for accum in range(1, 25):
        covered = 0
        for skip_every in (13, 17, 23, 97):
            # Skips every S steps leave runs of S-1 usable microbatches; if S-1 < accum
            # no window can close on its own. That case is handled by the stall bail-out
            # (test_stall_bailout_bounds_the_wait), not by the latch.
            if skip_every - 1 < accum:
                continue
            assert _latch_sim(accum, interval, 50000, latched=True, skip_every=skip_every), \
                f"no stop at accum={accum}, skip_every={skip_every}"
            covered += 1
        assert covered > 0, f"no non-degenerate skip pattern exercised for accum={accum}"
    print("PASS: latched stop still lands when microbatches take `continue` paths")


def test_stall_bailout_bounds_the_wait():
    """Repeated skips must not let a stop request hang until the epoch boundary.

    When failures resonate with the accumulation period (every accum-th microbatch
    skipped), no window ever closes. train_direct bounds that wait: after
    max(4*accum, 32) deferred microbatches it discards the partial window and stops.

    The poll is at the top of the body, so the request is still *seen* even when the
    skips cover every throttled poll step (accum=2 vs interval=10 is exactly that case).
    """
    interval = _sentinel_interval()
    for accum in (2, 5, 7, 9, 16):
        accum_count = 0
        event = False
        pending = False
        wait = 0
        stop_at = None
        for step in range(50000):
            if step % interval == 0:         # poll at the TOP, above the continue paths
                event = True
            if step % accum == 0:            # degenerate: resonant skip
                accum_count = 0
                continue
            accum_count += 1
            if accum_count >= accum:
                accum_count = 0
            if event:
                if pending:
                    wait += 1
                stall_limit = max(4 * accum, 32)
                if accum_count > 0 and wait <= stall_limit:
                    if not pending:
                        pending = True
                        wait = 0
                else:
                    stop_at = step
                    break
        assert stop_at is not None, f"stall bail-out never fired at accum={accum}"
        # Tight enough that doubling the stall limit would fail this.
        bound = interval + 2 * max(4 * accum, 32)
        assert stop_at <= bound, f"accum={accum}: stopped at {stop_at}, expected <= {bound}"
    # And the source must still carry the unthrottled epoch-boundary check as backstop.
    assert "stop_requested = soft_stop_event.is_set() or stop_sentinel.exists()" in _TRAIN_SRC
    print("PASS: stall bail-out bounds the wait under resonant skipped batches")


# =============================================== soft-stop / carry state machine (fix 2)


def _run_training(num_epochs, steps_per_epoch, accum, sentinel_from=None,
                  sentinel_until=None, interval=10, pre_fix=False, skip_every=0):
    """Faithful simulation of the soft-stop / accumulation-carry state machine.

    Mirrors train_direct.py: the epoch-start carry/reset, the microbatch accumulation,
    the latch + stop check + stall bail-out, the epoch-boundary flush (`defer_flush`)
    and the epoch-boundary stop handler.

    sentinel_from/until: absolute microbatch indices during which STOP_TRAINING exists
    (until=None means it is never removed).
    pre_fix=True reproduces the pre-fix form: no latch, no `soft_stop_carried` cap.
    """
    accum_count = 0
    pending = carried = early_exit = False
    wait = 0
    max_wait_seen = 0
    event = False
    carry_epochs, reset_epochs, saves = [], [], []
    epochs_entered = []
    absolute = 0
    stall_limit = max(4 * accum, 32)

    for epoch in range(num_epochs):
        epochs_entered.append(epoch)
        # NOTE: `wait` is deliberately NOT reset here - it is a function-level local in
        # train_direct.py and genuinely survives the epoch boundary carry. That is the
        # interaction this simulator exists to exercise.
        carry_accum = pending and accum_count > 0
        if carry_accum:
            carry_epochs.append(epoch)
        else:
            reset_epochs.append(epoch)
            accum_count = 0

        for step in range(steps_per_epoch):
            present = (sentinel_from is not None and absolute >= sentinel_from
                       and (sentinel_until is None or absolute < sentinel_until))
            absolute += 1
            # Poll at the top of the body (post-fix); pre-fix the throttled expression
            # was evaluated at the bottom as the stop condition itself.
            if not pre_fix and step % interval == 0 and present:
                event = True

            # A `continue` path (failed load / NaN loss / non-finite grad): zeroes the
            # window and skips the rest of the body, including the stop check.
            if skip_every and step % skip_every == 0:
                accum_count = 0
                continue

            accum_count += 1
            if accum_count >= accum:
                accum_count = 0

            stop_requested = (step % interval == 0 and present) if pre_fix else event

            if stop_requested:
                if pending:
                    wait += 1
                    max_wait_seen = max(max_wait_seen, wait)
                if accum_count > 0 and (pre_fix or wait <= stall_limit):
                    if not pending:
                        pending = True
                        wait = 0
                else:
                    accum_count = 0
                    saves.append(('mid', epoch, step + 1))
                    early_exit = True
                    break

        is_final = (epoch + 1 >= num_epochs)
        if pre_fix:
            defer_flush = pending and not is_final
        else:
            defer_flush = pending and not is_final and not carried
        if accum_count > 0 and not defer_flush:
            accum_count = 0

        boundary_present = (sentinel_from is not None and absolute >= sentinel_from
                            and (sentinel_until is None or absolute < sentinel_until))
        stop_requested = event or boundary_present
        if early_exit or stop_requested:
            if early_exit:
                break
            if accum_count > 0:
                pending = True
                carried = True
                continue
            saves.append(('boundary', epoch + 1, 0))
            break

    return {
        'saves': saves, 'carry_epochs': carry_epochs, 'reset_epochs': reset_epochs,
        'epochs_entered': epochs_entered, 'pending': pending, 'carried': carried,
        'max_wait_seen': max_wait_seen, 'stall_limit': stall_limit,
    }


def test_carry_state_machine():
    """Over a grid of shapes, a stop must always terminate with a checkpoint,
    and the running-total reset may be skipped at most once."""
    interval = _sentinel_interval()
    cases = 0
    for accum in (1, 2, 3, 5, 7, 8, 9, 12):
        for rem in range(accum):                      # steps_per_epoch % accum
            steps_per_epoch = 40 * accum + rem
            for stop_frac in (0.1, 0.97):             # early, and inside the tail window
                for num_epochs in (1, 2, 4):
                    for stop_epoch in range(num_epochs):
                        frm = stop_epoch * steps_per_epoch + int(steps_per_epoch * stop_frac)
                        r = _run_training(num_epochs, steps_per_epoch, accum,
                                          sentinel_from=frm, interval=interval)
                        cases += 1
                        assert r['saves'], (
                            f"no checkpoint written: accum={accum} rem={rem} "
                            f"spe={steps_per_epoch} ne={num_epochs} stop_epoch={stop_epoch}"
                        )
                        assert len(r['carry_epochs']) <= 1, (
                            f"accumulation carried {len(r['carry_epochs'])} times "
                            f"(accum={accum} rem={rem} ne={num_epochs}): running totals "
                            f"would never reset again"
                        )
                        last_epoch = r['epochs_entered'][-1]
                        assert last_epoch - stop_epoch <= 1, (
                            f"stop requested in epoch {stop_epoch} but ran through "
                            f"epoch {last_epoch}"
                        )
    print(f"PASS: carry state machine holds over {cases} shapes "
          f"(always checkpoints, carries at most once, stops within 1 epoch)")


def test_stall_counter_survives_the_boundary_carry():
    """`soft_stop_wait_steps` is a function-level local, so it survives the epoch
    boundary while `soft_stop_pending` stays True across a carry.

    The risk is a SPURIOUS bail-out: a wait accumulated in the tail of one epoch,
    carried into the next, tripping the limit during healthy training and discarding a
    live gradient window. It cannot happen, because a carry only occurs when the stop
    lands in the trailing partial window (at most accum-1 microbatches) and the carried
    window closes within accum more - both far below max(4*accum, 32). Asserted here
    rather than argued.
    """
    interval = _sentinel_interval()
    worst_ratio = 0.0
    carries = healthy = degenerate = 0
    for accum in (1, 2, 3, 5, 7, 8, 9, 12, 16, 32):
        for rem in range(1, max(2, accum)):          # non-zero remainder => partial tail
            steps_per_epoch = 40 * accum + rem
            for skip_every in (0, 11, 29):
                # Stop inside the trailing partial window, so the boundary carries it.
                frm = steps_per_epoch - rem
                r = _run_training(4, steps_per_epoch, accum, sentinel_from=frm,
                                  interval=interval, skip_every=skip_every)
                assert r['saves'], (
                    f"no checkpoint: accum={accum} rem={rem} skip_every={skip_every}"
                )
                assert len(r['carry_epochs']) <= 1, r['carry_epochs']
                carries += len(r['carry_epochs'])

                # Skips every S steps leave runs of S-1 microbatches. If S-1 < accum no
                # window can EVER close, so tripping the limit is the bail-out doing its
                # job, not a spurious fire. Only assert the no-spurious-fire property
                # where windows are actually able to close.
                if skip_every == 0 or skip_every - 1 >= accum:
                    healthy += 1
                    assert r['max_wait_seen'] <= r['stall_limit'], (
                        f"spurious bail-out: accum={accum} rem={rem} "
                        f"skip_every={skip_every} wait={r['max_wait_seen']} "
                        f"limit={r['stall_limit']}"
                    )
                    worst_ratio = max(worst_ratio, r['max_wait_seen'] / r['stall_limit'])
                else:
                    degenerate += 1
    assert carries > 0, "grid never actually produced a carry - assertions were vacuous"
    assert healthy and degenerate, "grid must cover both healthy and degenerate skips"
    print(f"PASS: stall counter survives the boundary carry with margin "
          f"(worst wait/limit = {worst_ratio:.3f} over {carries} carries; "
          f"{healthy} healthy / {degenerate} degenerate shapes)")


def test_briefly_present_sentinel():
    """A sentinel seen once by the poll, then removed.

    Pre-fix this was the run-corruption path: soft_stop_pending latched with nothing
    able to consume it, which permanently disabled the epoch-boundary flush and made
    carry_accum true every epoch, so running_loss / total_train_samples /
    processed_batches never reset again and avg_train_loss silently became a cumulative
    average over the rest of the run - while the run never actually stopped.

    Post-fix the request is honoured: the run checkpoints and exits.
    """
    interval = _sentinel_interval()
    # steps_per_epoch must NOT be divisible by accum, so a partial window survives the
    # boundary - that is what carry_accum feeds on.
    accum, steps_per_epoch, num_epochs = 8, 205, 5
    assert steps_per_epoch % accum != 0
    frm = 20  # coincides with a throttled poll step
    old = _run_training(num_epochs, steps_per_epoch, accum, sentinel_from=frm,
                        sentinel_until=frm + 1, interval=interval, pre_fix=True)
    new = _run_training(num_epochs, steps_per_epoch, accum, sentinel_from=frm,
                        sentinel_until=frm + 1, interval=interval)

    assert old['pending'], "pre-fix form should have latched soft_stop_pending"
    assert not old['saves'], "pre-fix form never actually wrote a checkpoint"
    assert len(old['carry_epochs']) >= num_epochs - 2, (
        f"expected the pre-fix carry corruption, got carry_epochs={old['carry_epochs']}"
    )

    assert new['saves'], "post-fix form must honour the request and checkpoint"
    assert len(new['carry_epochs']) <= 1, new['carry_epochs']
    assert len(new['epochs_entered']) == 1, (
        f"post-fix form should stop in the requesting epoch, ran {new['epochs_entered']}"
    )
    print("PASS: briefly-seen sentinel corrupts the pre-fix run; the fixed one "
          "checkpoints and exits")


def test_sentinel_never_polled_leaves_run_healthy():
    """A sentinel created and removed strictly between poll steps is never observed.

    Both forms must then run to completion with the running totals reset every epoch -
    i.e. no latch, no carry, no spurious stop.
    """
    interval = _sentinel_interval()
    accum, steps_per_epoch, num_epochs = 8, 205, 4
    frm = 21  # 21 % 10 != 0, and gone before the next poll at 30
    assert frm % interval != 0 and (frm + 1) % interval != 0
    for pre_fix in (True, False):
        r = _run_training(num_epochs, steps_per_epoch, accum, sentinel_from=frm,
                          sentinel_until=frm + 1, interval=interval, pre_fix=pre_fix)
        assert not r['saves'], f"pre_fix={pre_fix}: unobserved sentinel triggered a stop"
        assert r['carry_epochs'] == [], f"pre_fix={pre_fix}: {r['carry_epochs']}"
        assert r['reset_epochs'] == list(range(num_epochs)), (
            f"pre_fix={pre_fix}: running totals must reset every epoch"
        )
        assert len(r['epochs_entered']) == num_epochs
    print("PASS: sentinel removed between polls is never observed; run stays healthy")


# ============================================================== epoch-end predicates


def _should_validate(has_val_loader, eval_steps, epoch, start_epoch,
                     last_validation_step, global_step, force_validate):
    """Mirrors train_direct.py's should_validate expression."""
    return has_val_loader and (
        eval_steps == 0
        or (epoch == start_epoch and last_validation_step == 0)
        or force_validate
        or global_step - last_validation_step >= eval_steps
    )


def test_should_validate_predicate():
    ev = 11538
    # Fresh-seed paths (new run / phase transition / legacy ckpt / prepare_phase2) all
    # reach the loop with last_validation_step == 0.
    assert _should_validate(True, ev, 0, 0, 0, 0, False), "fresh run must validate epoch 0"
    # A resume no longer validates unconditionally...
    assert not _should_validate(True, ev, 7, 7, 200000, 200000, False)
    # ...but the cadence still fires one epoch later.
    assert _should_validate(True, ev, 8, 7, 200000, 200000 + 12499, False)
    # Zero-batch replayed epoch cannot advance global_step, so it is forced.
    assert _should_validate(True, ev, 14, 14, 200000, 200000, True)
    # eval_steps == 0 always validates; no val loader never does.
    assert _should_validate(True, 0, 3, 0, 999, 999, False)
    assert not _should_validate(False, 0, 0, 0, 0, 0, False)

    # With updates_per_epoch >= eval_steps, validation can never skip an epoch.
    for upe in (11538, 12499, 20000):
        last, g = 0, 0
        for epoch in range(1, 12):
            g += upe
            assert _should_validate(True, ev, epoch, 0, last, g, False), (
                f"skipped epoch {epoch} with updates_per_epoch={upe} >= eval_steps={ev}"
            )
            last = g
    print("PASS: should_validate keeps every fresh-seed path and cannot skip an epoch "
          "while updates_per_epoch >= eval_steps")


def test_zero_batch_epoch_carries_loss():
    """total_train_samples == 0 must never report a measured-nowhere 0.0."""
    def epoch_loss(running_loss, total_train_samples, checkpoint_train_loss):
        if total_train_samples > 0:
            return running_loss / total_train_samples, False
        return float(checkpoint_train_loss or 0.0), True

    loss, forced = epoch_loss(100.0, 50, 0.42)
    assert loss == 2.0 and not forced
    loss, forced = epoch_loss(0.0, 0, 0.42)
    assert loss == 0.42, "zero-batch epoch must carry the checkpoint's train_loss"
    assert forced, "zero-batch epoch must force validation"

    # AST: force_validate must actually be OR'd into the should_validate expression,
    # not merely assigned somewhere. A substring grep would pass on a dead assignment.
    sv_assigns = [
        n for n in ast.walk(_TRAIN_AST)
        if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "should_validate" for t in n.targets)
    ]
    assert sv_assigns, "should_validate assignment not found"
    assert any(
        any(isinstance(sub, ast.Name) and sub.id == "force_validate"
            for sub in ast.walk(a.value))
        for a in sv_assigns
    ), "force_validate is assigned but never read by should_validate"
    fv_assigns = [
        n for n in ast.walk(_TRAIN_AST)
        if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "force_validate" for t in n.targets)
    ]
    assert any(isinstance(a.value, ast.Constant) and a.value.value is False
               for a in fv_assigns), "force_validate must be reset to False each epoch"
    print("PASS: zero-batch epoch carries train_loss forward and forces validation")


def test_burn_in_gate():
    """An empty burn-in window must not reach np.max([]) / np.median([])."""
    import numpy as np

    def burn_in_end(vals, strategy="median"):
        if not vals:
            return None  # warn branch: leave best_metric alone
        if strategy == "last":
            baseline = float(vals[-1])
        elif strategy == "mean":
            baseline = float(np.mean(vals))
        elif strategy == "max":
            baseline = float(np.max(vals))
        else:
            baseline = float(np.median(vals))
        return max(baseline, float(np.max(vals)))

    assert burn_in_end([]) is None, "empty window must take the warn branch"
    assert burn_in_end([0.3, 0.5]) == 0.5
    assert burn_in_end([0.3, 0.5], "mean") == 0.5
    assert burn_in_end([0.9]) == 0.9
    assert "and not _burn_in_vals:" in _TRAIN_SRC, "empty burn-in guard missing"
    print("PASS: burn-in gate handles an empty window without numpy errors")


# ================================================= signal escalation contract (AST)


def test_escalation_not_triggered_by_sentinel_latch():
    """Behavioural model: the sentinel latch must not arm the hard-abort escalation."""
    soft_stop_event = threading.Event()
    _signal_seen = threading.Event()
    aborted = []

    def handler():
        if _signal_seen.is_set():
            aborted.append(True)
            return
        _signal_seen.set()
        soft_stop_event.set()

    soft_stop_event.set()   # STOP_TRAINING poll latches the stop first
    handler()               # operator's FIRST Ctrl+C -> must be graceful
    assert not aborted, "first signal hard-aborted because the sentinel had latched"
    handler()               # genuine second signal -> must escalate
    assert aborted == [True], "second signal failed to escalate"
    print("PASS: sentinel latch does not arm escalation; 2nd real signal still aborts")


def _handler_node():
    for node in ast.walk(_TRAIN_AST):
        if isinstance(node, ast.FunctionDef) and node.name == "_soft_stop_handler":
            return node
    raise AssertionError("_soft_stop_handler not found in train_direct.py")


def _is_call_on(node, obj, attr):
    return (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
            and node.func.attr == attr and isinstance(node.func.value, ast.Name)
            and node.func.value.id == obj)


def test_escalation_state_is_handler_owned():
    """AST assertion of the real invariant, robust to formatting.

    Two properties: the escalation branch tests `_signal_seen` and not
    `soft_stop_event`, and NOTHING outside the handler ever sets `_signal_seen`.
    """
    handler = _handler_node()
    lo, hi = handler.lineno, getattr(handler, "end_lineno", handler.lineno)

    tests = [n.test for n in ast.walk(handler) if isinstance(n, ast.If)]
    assert any(_is_call_on(t, "_signal_seen", "is_set") for t in tests), (
        "escalation branch must test the handler-owned _signal_seen event"
    )
    assert not any(_is_call_on(t, "soft_stop_event", "is_set") for t in tests), (
        "escalation must NOT test soft_stop_event - the STOP_TRAINING poll sets it too, "
        "so the operator's first Ctrl+C would hard-abort"
    )

    setters = [n for n in ast.walk(_TRAIN_AST) if _is_call_on(n, "_signal_seen", "set")]
    assert setters, "_signal_seen is never set"
    outside = [n.lineno for n in setters if not (lo <= n.lineno <= hi)]
    assert not outside, f"_signal_seen.set() called outside the handler at lines {outside}"

    # Nor may it be cleared, or rebound to a fresh Event, anywhere but its declaration -
    # either would silently re-arm the "this is the first signal" path.
    clears = [n.lineno for n in ast.walk(_TRAIN_AST)
              if _is_call_on(n, "_signal_seen", "clear")]
    assert not clears, f"_signal_seen.clear() would re-arm the graceful path: {clears}"
    rebinds = [n.lineno for n in ast.walk(_TRAIN_AST)
               if isinstance(n, ast.Assign)
               and any(isinstance(t, ast.Name) and t.id == "_signal_seen" for t in n.targets)]
    assert len(rebinds) == 1, f"_signal_seen must be bound exactly once, found {rebinds}"

    # The soft-stop poll must set soft_stop_event (the latch) and never _signal_seen.
    assert any(_is_call_on(n, "soft_stop_event", "set") and not (lo <= n.lineno <= hi)
               for n in ast.walk(_TRAIN_AST)), "sentinel latch must set soft_stop_event"
    print("PASS: escalation state is handler-owned (AST-verified)")


# ================================================================= TrainingState


def test_training_state_roundtrip():
    from training_utils import TrainingState
    ts = TrainingState(epoch=4, global_step=1234, completed_epochs=3,
                       burn_in_values=[0.1, 0.2], last_validation_step=999)
    d = ts.to_dict()
    assert d['burn_in_values'] == [0.1, 0.2]
    assert d['last_validation_step'] == 999
    assert d['completed_epochs'] == 3
    back = TrainingState.from_dict(d)
    assert back.burn_in_values == [0.1, 0.2]
    assert back.last_validation_step == 999

    legacy = TrainingState.from_dict({'epoch': 2, 'global_step': 7, 'removed_field': 1})
    assert legacy.burn_in_values == []
    assert legacy.last_validation_step == 0
    assert legacy.epoch == 2

    a, b = TrainingState(), TrainingState()
    a.burn_in_values.append(1.0)
    assert b.burn_in_values == [], "default_factory must not share a list"
    print("PASS: TrainingState round-trips new fields and tolerates legacy checkpoints")


TESTS = [
    test_sampler_offset,
    test_sampler_offset_abandoned,
    test_len_semantics_against_real_dataloader,
    test_get_state_reports_pending_offset_only,
    test_resume_offset_is_not_double_applied_after_a_batch_size_change,
    test_resume_offset_unchanged_batch_size_is_untouched,
    test_sample_in_epoch_records_the_true_position,
    test_fallback_is_gated_on_the_applied_flag_not_start_step,
    test_retarget_rescales_lr_and_completes_the_anneal,
    test_effective_batch_size_is_never_shadowed,
    test_retarget_syncs_initial_lr_into_the_optimizer,
    test_retarget_preserves_per_group_lr_ratios,
    test_guard_catches_min_lr_and_gamma_drift,
    test_min_lr_ordering_inside_retarget,
    test_guard_arming_line_is_branch_correct,
    test_effective_batch_guard_branches,
    test_lr_schedule_guard_fires_on_dataset_size_drift,
    test_lr_schedule_guard_peak_lr_axis,
    test_lr_schedule_guard_is_disarmed_without_restored_state,
    test_effective_batch_policy_is_wired,
    test_stop_reachability,
    test_stop_survives_skipped_batches,
    test_stall_bailout_bounds_the_wait,
    test_carry_state_machine,
    test_stall_counter_survives_the_boundary_carry,
    test_briefly_present_sentinel,
    test_sentinel_never_polled_leaves_run_healthy,
    test_should_validate_predicate,
    test_zero_batch_epoch_carries_loss,
    test_burn_in_gate,
    test_escalation_not_triggered_by_sentinel_latch,
    test_escalation_state_is_handler_owned,
    test_training_state_roundtrip,
]

if __name__ == "__main__":
    for t in TESTS:
        t()
    print(f"\nALL {len(TESTS)} CHECKS PASSED")
