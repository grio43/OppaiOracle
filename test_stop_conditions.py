"""Tests for the training stop system (``stop_conditions.py``).

Run directly:  python test_stop_conditions.py

The two that matter most:

  * :func:`test_v1_p2_replay_does_not_stop` — the real V1-P2 val/mAP sequence must
    NOT trip any rule. That run was manually stopped at 6 of 15 planned epochs
    while val/mAP rose monotonically every epoch and val_loss fell every epoch.
    Any rule set that stops it is reproducing the original defect.
  * :func:`test_soft_stop_resume_is_identical` — a history persisted, truncated at
    a soft stop, restored, and replayed must yield the same verdict as an
    uninterrupted run. This is the property the whole module is built for.
"""
import json
import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))

import stop_conditions as sc


# Real V1 Phase-2 measurements, from TRAINING_HEALTH_TRACKER.md's per-epoch table
# (archive epochs E0-E5, 0-based; the table's 1-based log epochs are E+1).
V1_P2_MAP = [0.659055, 0.663135, 0.667649, 0.669132, 0.672210, 0.674373]
V1_P2_VAL_LOSS = [0.000464, 0.000461, 0.000458, 0.000455, 0.000453, 0.000451]
V1_P2_TRAIN_LOSS = [0.000495, 0.000490, 0.000489, 0.000487, 0.000485, 0.000483]


def _history(values, val_losses=None, train_losses=None, start_epoch=1, fingerprints=None,
             clean_margins=None):
    """Build a history the way train_direct does, one record per validated epoch."""
    history = []
    for i, value in enumerate(values):
        history = sc.append_record(history, sc.make_record(
            epoch=start_epoch + i,
            global_step=(start_epoch + i) * 1000,
            selection_value=value,
            val_mAP=value,
            val_loss=val_losses[i] if val_losses else 0.001 - i * 1e-6,
            train_loss=train_losses[i] if train_losses else 0.002 - i * 1e-6,
            val_f1_macro=0.0,
            val_f1_micro=0.0,
            lr=1e-5,
            clean_margin=clean_margins[i] if clean_margins else None,
            fingerprints=fingerprints[i] if fingerprints else None,
        ))
    return history


def _policy(**kw):
    base = dict(mode="halt", min_delta=3e-3, window=3, confirm=2, min_epochs=4,
                burn_in_epochs=0, regression_delta=6e-3, regression_confirm=2,
                overfit_confirm=3, max_epochs=15)
    base.update(kw)
    return sc.StopPolicy(**base)


# --------------------------------------------------------------------------
# The regression that started all of this
# --------------------------------------------------------------------------

def test_v1_p2_replay_does_not_stop():
    """The real V1-P2 curve must survive every rule."""
    history = _history(V1_P2_MAP, V1_P2_VAL_LOSS, V1_P2_TRAIN_LOSS)
    verdict = sc.evaluate(history, _policy(), completed_epochs=6)
    assert verdict.action == "continue", verdict.summary_line()
    assert not verdict.stop_triggers, verdict.reasons()

    # At n=6 with confirm=2 the plateau rule is still `pending` — it needs 7
    # validated epochs to confirm. `pending` is a distinct state from `ok` on
    # purpose: "not enough measurement to judge" must never read as "no
    # improvement", which is how the old code turned skipped validations into
    # patience.
    plateau = next(t for t in verdict.triggers if t.name == "plateau")
    assert plateau.state == "pending", plateau.detail

    # The single window that IS evaluable clears the noise floor with room to
    # spare — the number canary #1's >=1.01x/epoch band was blind to.
    gain = sc._window_gain(V1_P2_MAP, len(V1_P2_MAP), 3)
    assert math.isclose(gain, max(V1_P2_MAP[3:6]) - max(V1_P2_MAP[0:3]), rel_tol=1e-9)
    assert gain > 2 * 3e-3, f"window gain {gain} should clear min_delta twice over"
    at_confirm_1 = sc.evaluate(history, _policy(confirm=1), completed_epochs=6)
    assert next(t for t in at_confirm_1.triggers if t.name == "plateau").state == "ok"
    assert at_confirm_1.action == "continue"

    # The overfitting rule must be quiet: val_loss fell monotonically alongside
    # train_loss for the whole run. V1 stopped to avoid overfitting; there was no
    # overfitting signal at any point.
    overfit = next(t for t in verdict.triggers if t.name == "overfit_loss")
    assert overfit.state == "ok" and not overfit.tripped, overfit.detail


def test_naive_per_epoch_patience_accrues_on_the_healthy_run():
    """Why the plateau test is window-over-window rather than per-epoch.

    V1-P2's per-epoch mAP gains straddle the 3e-3 resolution floor (+0.0041,
    +0.0045, +0.0015, +0.0031, +0.0022), so a rule asking "did THIS epoch beat
    the best by more than min_delta" scores 2 of 5 monotonically-improving epochs
    as non-improvements — enough to trip patience=2. Taking the best of a 3-epoch
    window instead accumulates the same real progress into one comparison that
    clears the floor twice over.
    """
    gains = [b - a for a, b in zip(V1_P2_MAP, V1_P2_MAP[1:])]
    assert all(g > 0 for g in gains), "the curve is monotonically improving"
    below_floor = [g for g in gains if g < 3e-3]
    assert len(below_floor) == 2, gains

    best, longest_run, run = V1_P2_MAP[0], 0, 0
    for value in V1_P2_MAP[1:]:
        if value > best + 3e-3:
            best, run = value, 0
        else:
            run += 1
            longest_run = max(longest_run, run)
    assert longest_run >= 1, "per-epoch testing accrues patience on this curve"

    # Window-over-window on the same data and the same min_delta: one clean pass.
    assert sc._window_gain(V1_P2_MAP, len(V1_P2_MAP), 3) > 2 * 3e-3


# --------------------------------------------------------------------------
# Soft-stop / resume equivalence — the reason nothing here is a counter
# --------------------------------------------------------------------------

def test_soft_stop_resume_is_identical():
    """Persist -> truncate -> restore -> replay must equal the uninterrupted run."""
    values = [0.60, 0.62, 0.64, 0.6405, 0.6408, 0.6410, 0.6411, 0.6412]
    policy = _policy(max_epochs=20)

    uninterrupted = sc.evaluate(_history(values), policy, completed_epochs=len(values))

    # A soft stop after epoch 5: the checkpoint holds 5 records. The resumed
    # process rebuilds from them and continues.
    persisted = _history(values[:5])
    round_tripped = json.loads(json.dumps(persisted))  # what the checkpoint does
    resumed = round_tripped
    for i, value in enumerate(values[5:], start=6):
        resumed = sc.append_record(resumed, sc.make_record(
            epoch=i, global_step=i * 1000, selection_value=value, val_mAP=value,
            val_loss=0.001 - (i - 1) * 1e-6, train_loss=0.002 - (i - 1) * 1e-6,
            val_f1_macro=0.0, val_f1_micro=0.0, lr=1e-5,
        ))
    after_resume = sc.evaluate(resumed, policy, completed_epochs=len(values))

    assert after_resume.action == uninterrupted.action
    assert after_resume.n_validations == uninterrupted.n_validations
    assert after_resume.epochs_since_best == uninterrupted.epochs_since_best
    assert after_resume.to_dict()["triggers"] == uninterrupted.to_dict()["triggers"]


def test_replayed_epoch_is_not_double_counted():
    """A mid-epoch soft stop replays an epoch; the record must be replaced.

    This is the failure mode a counter cannot avoid. The old code incremented
    ``patience_counter`` per epoch, so a replayed epoch either advanced it twice
    or (on the other resume path) reset it to zero.
    """
    history = _history([0.60, 0.62, 0.64])
    assert len(history) == 3
    # Resume replays epoch 3 and re-validates it, landing a slightly different
    # value (different flip seeds / dropout are off, but batch composition can
    # differ via the error filter).
    history = sc.append_record(history, sc.make_record(
        epoch=3, global_step=3000, selection_value=0.6401, val_mAP=0.6401,
        val_loss=0.001, train_loss=0.002, val_f1_macro=0.0, val_f1_micro=0.0, lr=1e-5,
    ))
    assert len(history) == 3, "replayed epoch must replace, not append"
    assert [r["epoch"] for r in history] == [1, 2, 3]
    assert history[-1]["selection_value"] == 0.6401


def test_out_of_order_append_is_sorted():
    history = sc.append_record(_history([0.6, 0.62]), sc.make_record(
        epoch=1, global_step=1, selection_value=0.61, val_mAP=0.61, val_loss=0.001,
        train_loss=0.002, val_f1_macro=0.0, val_f1_micro=0.0, lr=1e-5,
    ))
    assert [r["epoch"] for r in history] == [1, 2]
    assert history[0]["selection_value"] == 0.61


# --------------------------------------------------------------------------
# Rules
# --------------------------------------------------------------------------

def test_plateau_trips_on_a_real_plateau():
    # Rises, then flattens well inside the noise floor.
    values = [0.60, 0.63, 0.66, 0.6601, 0.6602, 0.6601, 0.6603, 0.6602]
    verdict = sc.evaluate(_history(values), _policy(), completed_epochs=len(values))
    plateau = next(t for t in verdict.triggers if t.name == "plateau")
    assert plateau.tripped, plateau.detail
    assert verdict.action == "halt"


def test_plateau_needs_confirmation():
    """One sub-threshold window is not enough; `confirm` consecutive ones are."""
    # 6 epochs: exactly enough for ONE window evaluation, so confirm=2 cannot
    # be satisfied yet even though that single gain is below min_delta.
    values = [0.660, 0.661, 0.662, 0.6621, 0.6622, 0.6623]
    verdict = sc.evaluate(_history(values), _policy(confirm=2), completed_epochs=6)
    plateau = next(t for t in verdict.triggers if t.name == "plateau")
    assert plateau.state == "pending", plateau.detail
    assert verdict.action == "continue"
    # With confirm=1 the same history trips.
    verdict1 = sc.evaluate(_history(values), _policy(confirm=1), completed_epochs=6)
    assert next(t for t in verdict1.triggers if t.name == "plateau").tripped


def test_min_epochs_floor_blocks_everything():
    values = [0.60, 0.55, 0.50, 0.45, 0.40, 0.35, 0.30]
    verdict = sc.evaluate(_history(values), _policy(min_epochs=99), completed_epochs=7)
    assert verdict.action == "continue"
    assert all(t.state == "pending" for t in verdict.triggers
               if t.name in {"plateau", "peak_decline", "overfit_loss"})


def test_burn_in_epochs_are_excluded():
    """Burn-in records must not enter the stop history at all."""
    # A wild burn-in spike followed by a healthy climb.
    values = [0.90, 0.10, 0.60, 0.63, 0.66, 0.69, 0.72, 0.75]
    verdict = sc.evaluate(_history(values), _policy(burn_in_epochs=2), completed_epochs=8)
    assert verdict.n_validations == 6
    assert verdict.best_value == 0.75
    assert verdict.action == "continue"
    # Without the exclusion the 0.90 spike is the best forever, which is exactly
    # the bar-too-high failure the old burn-in `max()` produced.
    naive = sc.evaluate(_history(values), _policy(burn_in_epochs=0), completed_epochs=8)
    assert naive.best_value == 0.90


def test_peak_decline_trips_on_sustained_drop():
    values = [0.60, 0.65, 0.70, 0.72, 0.70, 0.69]
    verdict = sc.evaluate(_history(values), _policy(regression_confirm=2), completed_epochs=6)
    decline = next(t for t in verdict.triggers if t.name == "peak_decline")
    assert decline.tripped, decline.detail
    assert decline.values["best"] == 0.72


def test_peak_decline_ignores_a_single_dip():
    """A one-epoch dip inside the noise band is not a decline.

    V1-P2's E5 posted an f1_macro dip (0.013124 -> 0.013035) while val/mAP rose.
    Under the old rule that was a patience increment; here a lone dip is nothing.
    """
    values = [0.60, 0.65, 0.70, 0.699, 0.71]
    verdict = sc.evaluate(_history(values), _policy(), completed_epochs=5)
    decline = next(t for t in verdict.triggers if t.name == "peak_decline")
    assert not decline.tripped, decline.detail


def test_clean_margin_never_demotes_a_decline():
    """A wrong positive can be memorized while its sibling margin rises."""
    declining = [0.60, 0.65, 0.70, 0.72, 0.70, 0.69]
    for margins in (None, [0.1] * 6, [0.10, 0.12, 0.14, 0.16, 0.18, 0.20]):
        verdict = sc.evaluate(_history(declining, clean_margins=margins),
                              _policy(), completed_epochs=6)
        decline = next(t for t in verdict.triggers if t.name == "peak_decline")
        assert decline.tripped and not decline.advisory, decline.detail
        assert decline in verdict.stop_triggers
        assert verdict.action == "halt", verdict.summary_line()


def test_stale_clean_margin_cannot_demote_a_current_decline():
    """A hole in the recent record tail must not let OLD margins demote.

    Regression: clean_margin() used to filter out records lacking the metric
    and take the last `span` OF THE SURVIVORS, so when recent epochs carried
    no margin (telemetry hiccup / sibling_groups_path unresolved on resume)
    the demotion window covered epochs OLDER than peak_decline's window. A
    genuine current decline could then be demoted on stale evidence.
    """
    declining = [0.60, 0.65, 0.70, 0.72, 0.70, 0.69]
    # Margins only on the four OLDEST epochs; the two most recent lack one.
    stale_then_none = [0.10, 0.12, 0.14, 0.16, None, None]
    verdict = sc.evaluate(_history(declining, clean_margins=stale_then_none),
                          _policy(), completed_epochs=6)
    arbiter = next(t for t in verdict.triggers if t.name == "clean_margin")
    assert arbiter.state == "pending", arbiter.detail
    decline = next(t for t in verdict.triggers if t.name == "peak_decline")
    assert decline.tripped and not decline.advisory, decline.detail
    assert "DEMOTED" not in decline.detail
    assert verdict.action == "halt"


def test_clean_margin_corroborates_a_real_decline():
    declining = [0.60, 0.65, 0.70, 0.72, 0.70, 0.69]
    falling_clean = [0.20, 0.19, 0.18, 0.17, 0.14, 0.11]
    verdict = sc.evaluate(_history(declining, clean_margins=falling_clean),
                          _policy(), completed_epochs=6)
    decline = next(t for t in verdict.triggers if t.name == "peak_decline")
    assert decline.tripped and not decline.advisory, decline.detail
    assert "CORROBORATED" in decline.detail
    assert verdict.action == "halt"


def test_missing_clean_margin_does_not_silently_corroborate():
    """No arbiter must read as `pending`, never as agreement."""
    declining = [0.60, 0.65, 0.70, 0.72, 0.70, 0.69]
    verdict = sc.evaluate(_history(declining), _policy(), completed_epochs=6)
    arbiter = next(t for t in verdict.triggers if t.name == "clean_margin")
    assert arbiter.state == "pending", arbiter.detail
    assert "asl_telemetry" in arbiter.detail  # tells the operator how to get it
    decline = next(t for t in verdict.triggers if t.name == "peak_decline")
    assert decline.tripped and not decline.advisory
    assert "CORROBORATED" not in decline.detail
    assert verdict.action == "halt"


def test_clean_margin_does_not_stop_on_its_own():
    """Four confusable groups out of 19.3K tags cannot end a run."""
    flat = [0.700, 0.7005, 0.7002, 0.7004, 0.7003, 0.7006]
    collapsing_clean = [0.30, 0.25, 0.20, 0.15, 0.10, 0.05]
    verdict = sc.evaluate(_history(flat, clean_margins=collapsing_clean),
                          _policy(min_epochs=99), completed_epochs=6)
    assert verdict.action == "continue"
    assert not verdict.stop_triggers
    arbiter = next(t for t in verdict.triggers if t.name == "clean_margin")
    assert not arbiter.tripped
    assert arbiter.values["change"] < 0  # reported, not acted on


def test_per_phase_geometry_override():
    """A short phase must be able to use a tighter geometry, and phase 3 off."""
    from Configuration_System import load_config

    config = load_config(REPO / "configs" / "unified_config.yaml", validate=True)

    config.training.phase = 1
    p1 = sc.StopPolicy.from_config(config.training)
    assert (p1.window, p1.confirm) == (3, 2), (p1.window, p1.confirm)

    config.training.phase = 2
    p2 = sc.StopPolicy.from_config(config.training)
    assert (p2.window, p2.confirm) == (2, 1), (p2.window, p2.confirm)
    # The point of the override: a short phase gets usable coverage.
    assert p2.first_confirmable_epoch() < p1.first_confirmable_epoch()

    config.training.phase = 3
    p3 = sc.StopPolicy.from_config(config.training)
    assert p3.mode == "off", p3.mode
    # ...and 'off' cannot stop a 3-epoch phase no matter what the curve does.
    flat = [0.70, 0.70, 0.70, 0.70, 0.70, 0.70, 0.70, 0.70]
    assert sc.evaluate(_history(flat), p3, completed_epochs=8).action == "continue"


def test_phase_override_inherits_unset_keys():
    class _Phase:
        mode = None
        window = 2
        confirm = None
        min_epochs = None

    class _Phases:
        phase1 = _Phase()

    class _Cfg:
        phase = 1
        early_stopping_phases = _Phases()
        early_stopping_mode = "halt"
        early_stopping_window = 5
        early_stopping_confirm = 4
        early_stopping_min_epochs = 6
        early_stopping_min_delta = 3e-3
        early_stopping_burn_in_epochs = 2
        early_stopping_regression_delta = 6e-3
        early_stopping_regression_confirm = 2
        early_stopping_overfit_confirm = 3
        num_epochs = 40

    policy = sc.StopPolicy.from_config(_Cfg())
    assert policy.window == 2          # overridden
    assert policy.confirm == 4         # inherited
    assert policy.mode == "halt"       # inherited
    assert policy.min_epochs == 6      # inherited


def test_overfit_rule_needs_both_directions():
    rising_val = [0.001, 0.002, 0.003, 0.004]
    falling_train = [0.004, 0.003, 0.002, 0.001]
    policy = _policy(overfit_confirm=3)
    trig = sc.overfit_loss(rising_val, falling_train, policy)
    assert trig.tripped, trig.detail
    # Both rising = instability or a data incident, not overfitting.
    trig2 = sc.overfit_loss(rising_val, rising_val, policy)
    assert not trig2.tripped, trig2.detail


def test_overfit_rule_fires_within_the_full_evaluate():
    values = [0.70, 0.70, 0.70, 0.70, 0.70]
    history = _history(
        values,
        val_losses=[0.0010, 0.0011, 0.0012, 0.0013, 0.0014],
        train_losses=[0.0020, 0.0019, 0.0018, 0.0017, 0.0016],
    )
    verdict = sc.evaluate(history, _policy(overfit_confirm=3), completed_epochs=5)
    assert next(t for t in verdict.triggers if t.name == "overfit_loss").tripped
    assert verdict.action == "halt"


# --------------------------------------------------------------------------
# Modes, hygiene, advisories
# --------------------------------------------------------------------------

def test_advise_mode_never_halts():
    values = [0.60, 0.63, 0.66, 0.6601, 0.6602, 0.6601, 0.6603, 0.6602]
    verdict = sc.evaluate(_history(values), _policy(mode="advise"), completed_epochs=8)
    assert verdict.stop_triggers
    assert verdict.action == "advise"


def test_off_mode_still_reports():
    values = [0.60, 0.63, 0.66, 0.6601, 0.6602, 0.6601, 0.6603, 0.6602]
    verdict = sc.evaluate(_history(values), _policy(mode="off"), completed_epochs=8)
    assert verdict.action == "continue"
    # The rules still evaluated; only the action is suppressed.
    assert next(t for t in verdict.triggers if t.name == "plateau").tripped


def test_non_finite_values_are_dropped_not_propagated():
    history = _history([0.60, 0.62, 0.64, 0.66])
    history = sc.append_record(history, sc.make_record(
        epoch=5, global_step=5000, selection_value=float("nan"), val_mAP=float("nan"),
        val_loss=0.001, train_loss=0.002, val_f1_macro=0.0, val_f1_micro=0.0, lr=1e-5,
    ))
    verdict = sc.evaluate(history, _policy(), completed_epochs=5)
    assert verdict.n_validations == 4
    assert verdict.best_value == 0.66


def test_budget_trigger_reports_position_and_never_trips():
    verdict = sc.evaluate(_history(V1_P2_MAP), _policy(max_epochs=15), completed_epochs=6)
    b = next(t for t in verdict.triggers if t.name == "budget")
    assert not b.tripped
    assert b.values["remaining"] == 9.0
    assert "6/15" in b.detail


def _falling_fp(name, values):
    return [{name: v} for v in values]


def test_fingerprint_advisory_is_suppressed_while_the_metric_improves():
    """The failure mode that mattered: LR decay moves all three fingerprints in
    the problem direction during healthy training (canary #7: "Healthy: monotonic
    decline as logits tighten with LR decay"). Flagging on direction alone would
    fire on every good run — canary #1's mistake, one layer down."""
    fps = _falling_fp("mean_sigmoid_topK_unlabeled", [0.50, 0.45, 0.40, 0.35, 0.30])
    improving = _history([0.60, 0.62, 0.64, 0.66, 0.68], fingerprints=fps)
    verdict = sc.evaluate(improving, _policy(), completed_epochs=5)
    assert not verdict.advisories, [t.detail for t in verdict.advisories]


def test_fingerprint_advisory_fires_when_the_metric_has_stalled():
    fps = _falling_fp("mean_sigmoid_topK_unlabeled", [0.50, 0.45, 0.40, 0.35, 0.30])
    stalled = _history([0.680, 0.680, 0.6801, 0.6800, 0.6802], fingerprints=fps)
    verdict = sc.evaluate(stalled, _policy(), completed_epochs=5)
    assert verdict.advisories, "should confirm a suspect pattern"
    assert all(t.advisory for t in verdict.advisories)
    # An advisory must never change the action, even when it fires.
    assert not verdict.stop_triggers
    assert verdict.action == "continue"


def test_pred_pos_ratio_needs_to_cross_its_level_not_merely_fall():
    """Cole's EPR is about the ratio dropping BELOW ~1.0, not about it dropping.

    A model emitting 3x the labelled positives and tightening toward 1.5x is
    calibrating, not collapsing. Encoding that row as direction-only was wrong.
    """
    stalled = [0.680, 0.680, 0.6801, 0.6800, 0.6802]
    above = _history(stalled, fingerprints=_falling_fp("pred_pos_ratio", [3.0, 2.5, 2.0, 1.6, 1.3]))
    assert not sc.evaluate(above, _policy(), completed_epochs=5).advisories
    below = _history(stalled, fingerprints=_falling_fp("pred_pos_ratio", [1.4, 1.2, 1.0, 0.8, 0.6]))
    flagged = sc.evaluate(below, _policy(), completed_epochs=5).advisories
    assert [t.name for t in flagged] == ["fingerprint:pred_pos_ratio"], flagged


def test_overfit_rule_refuses_to_span_a_loss_shape_change():
    """val_loss is not comparable across an ASL gamma_neg step.

    Gamma steps are manual (stop -> set override -> resume), so a straddling
    window is exactly what a resumed run evaluates first.
    """
    rising_val = [0.001, 0.002, 0.003, 0.004]
    falling_train = [0.004, 0.003, 0.002, 0.001]
    policy = _policy(overfit_confirm=3)
    same = sc.overfit_loss(rising_val, falling_train, policy, loss_ids=[7.0, 7.0, 7.0, 7.0])
    assert same.tripped, same.detail
    straddling = sc.overfit_loss(rising_val, falling_train, policy, loss_ids=[7.0, 7.0, 6.0, 6.0])
    assert straddling.state == "pending", straddling.detail
    assert "loss-shape change" in straddling.detail


def test_loss_id_flows_through_evaluate():
    history = _history([0.70] * 5,
                       val_losses=[0.0010, 0.0011, 0.0012, 0.0013, 0.0014],
                       train_losses=[0.0020, 0.0019, 0.0018, 0.0017, 0.0016])
    for i, record in enumerate(history):
        record["loss_id"] = 7.0 if i < 2 else 6.0
    verdict = sc.evaluate(history, _policy(overfit_confirm=3), completed_epochs=5)
    trig = next(t for t in verdict.triggers if t.name == "overfit_loss")
    assert trig.state == "pending", trig.detail
    # Same data, one loss shape: it trips.
    for record in history:
        record["loss_id"] = 7.0
    trig2 = next(t for t in sc.evaluate(history, _policy(overfit_confirm=3), completed_epochs=5).triggers
                 if t.name == "overfit_loss")
    assert trig2.tripped, trig2.detail


def test_every_verdict_reports_the_full_trigger_set():
    """Guard against a rule silently disappearing from evaluate().

    Caught a real regression: the `budget` trigger was dropped by an edit and
    nothing else noticed, because a missing trigger cannot fail an
    `any(t.tripped)` check — it just stops being reported.
    """
    expected = {"plateau", "peak_decline", "overfit_loss", "budget", "clean_margin"}
    for label, verdict in (
        ("short history", sc.evaluate(_history([0.6, 0.62]), _policy(), completed_epochs=2)),
        ("full history", sc.evaluate(_history(V1_P2_MAP), _policy(), completed_epochs=6)),
        ("mode off", sc.evaluate(_history(V1_P2_MAP), _policy(mode="off"), completed_epochs=6)),
        ("below floor", sc.evaluate(_history(V1_P2_MAP), _policy(min_epochs=99), completed_epochs=6)),
    ):
        names = {t.name for t in verdict.triggers}
        assert expected <= names, (label, sorted(expected - names))


def test_verdict_is_json_serializable():
    verdict = sc.evaluate(_history(V1_P2_MAP), _policy(), completed_epochs=6)
    text = json.dumps(verdict.to_dict())
    assert json.loads(text)["action"] == "continue"


def test_policy_from_config_reads_the_real_config():
    """The live YAML must produce a usable policy and pass its own validation."""
    from Configuration_System import load_config

    config = load_config(REPO / "configs" / "unified_config.yaml", validate=True)
    policy = sc.StopPolicy.from_config(config.training)
    assert policy.mode in {"advise", "halt", "off"}
    assert policy.min_delta > 0
    assert policy.window >= 1
    assert policy.max_epochs == int(config.training.num_epochs)
    # The plateau rule must be able to confirm inside the epoch budget.
    needed = policy.burn_in_epochs + 2 * policy.window + policy.confirm - 1
    assert policy.max_epochs >= needed, (policy.max_epochs, needed)
    # And the live config must not stop the V1-P2 curve.
    verdict = sc.evaluate(_history(V1_P2_MAP, V1_P2_VAL_LOSS, V1_P2_TRAIN_LOSS),
                          policy, completed_epochs=6)
    assert not verdict.stop_triggers, verdict.reasons()


def test_training_state_round_trips_the_history():
    """eval_history must survive TrainingState -> dict -> TrainingState."""
    from training_utils import TrainingState

    history = _history(V1_P2_MAP)
    state = TrainingState(eval_history=[dict(r) for r in history],
                          frozen_macro_tag_indices=[1, 2, 3],
                          stop_advisories_seen=["4:plateau"])
    restored = TrainingState.from_dict(state.to_dict())
    assert restored.eval_history == history
    assert restored.frozen_macro_tag_indices == [1, 2, 3]
    assert restored.stop_advisories_seen == ["4:plateau"]
    # And a legacy checkpoint with none of these fields must still load.
    legacy = TrainingState.from_dict({"epoch": 3, "best_metric": 0.5})
    assert legacy.eval_history == []
    assert legacy.frozen_macro_tag_indices == []


# --------------------------------------------------------------------------
# Wiring invariants in train_direct.py (source/AST-locked, no training run)
# --------------------------------------------------------------------------

_TRAIN_SRC = (REPO / "train_direct.py").read_text(encoding="utf-8")

# AST, not text search: both checks below must survive the explanatory comments
# that name the very constructs they forbid.
import ast as _ast

_TRAIN_AST = _ast.parse(_TRAIN_SRC)


def test_no_accumulated_patience_counter_remains():
    """`patience_counter` must never be incremented again.

    An accumulated counter is what broke across soft stops and what advanced on
    epochs that produced no measurement. It is now derived from eval_history.
    """
    bad = [
        node for node in _ast.walk(_TRAIN_AST)
        if isinstance(node, _ast.AugAssign)
        and isinstance(node.target, _ast.Attribute)
        and node.target.attr == "patience_counter"
    ]
    assert not bad, [node.lineno for node in bad]


def test_lr_ratio_gate_is_gone():
    """The `lr / scheduler.max_lr < 0.5` gate must not gate stopping again.

    On num_cycles=1 it is false until the midpoint of the anneal, so it silently
    suppressed the entire mechanism for the first half of every run.
    """
    bad = [
        node.lineno for node in _ast.walk(_TRAIN_AST)
        if isinstance(node, _ast.Name) and node.id in {"lr_ratio", "cycle_max_lr"}
    ]
    assert not bad, bad


def test_history_append_is_guarded_by_should_validate():
    """A stop rule must never see an epoch that did not validate."""
    import ast

    tree = ast.parse(_TRAIN_SRC)

    def _calls_append(node):
        for sub in ast.walk(node):
            if (isinstance(sub, ast.Call) and isinstance(sub.func, ast.Attribute)
                    and sub.func.attr == "append_record"):
                return True
        return False

    total = sum(1 for n in ast.walk(tree)
                if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                and n.func.attr == "append_record")
    assert total == 1, f"expected exactly one append_record call, found {total}"

    guarded = False
    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        if isinstance(node.test, ast.Name) and node.test.id == "should_validate":
            if any(_calls_append(stmt) for stmt in node.body):
                guarded = True
                break
    assert guarded, "append_record is not inside an `if should_validate:` body"


def test_burn_in_baseline_uses_the_strategy_not_the_max():
    """The burn-in bar must be the strategy's summary.

    The old line was `max(baseline, best_during_burnin)`, which is always the
    max — making `median`/`mean`/`last` dead options and setting the post-burn-in
    bar to the burn-in high-water mark.
    """
    assert "training_state.best_metric = baseline" in _TRAIN_SRC
    assert "max(baseline, best_during_burnin)" not in _TRAIN_SRC


def test_history_is_mirrored_into_training_state():
    """Every checkpoint written after the append must carry the history."""
    assert "training_state.eval_history = " in _TRAIN_SRC


def test_only_halt_mode_breaks_the_epoch_loop():
    """Legacy non-WSD advice must run to the budget; WSD completion is handled separately."""
    import re
    block = _TRAIN_SRC[_TRAIN_SRC.index("# Act on the verdict."):]
    block = block[:block.index("# Clear the one-shot SAVE_CHECKPOINT sentinel")]
    assert "if stop_verdict.action == 'halt':" in block
    # Exactly one `break`, and it sits under the halt branch.
    assert len(re.findall(r"^\s*break\s*$", block, re.M)) == 1
    assert block.index("action == 'halt'") < block.index("break")


def _main():
    tests = [(name, obj) for name, obj in sorted(globals().items())
             if name.startswith("test_") and callable(obj)]
    failures = []
    for name, fn in tests:
        try:
            fn()
        except Exception as exc:  # noqa: BLE001
            failures.append((name, exc))
            print(f"FAIL {name}: {exc}")
        else:
            print(f"ok   {name}")
    print(f"\n{len(tests) - len(failures)}/{len(tests)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(_main())
