"""Training stop conditions — history-derived, and correct across a soft stop.

Every rule in this module is a **pure function of the persisted per-epoch
evaluation history** (``TrainingState.eval_history``). Nothing here is an
incrementally-mutated counter. That is the whole design, and it is what makes
the system correct across a soft stop: a resumed process rebuilds each verdict
from the same history the pre-stop process had, so a stop decision cannot
drift, double-count a replayed epoch, or silently reset to zero.

The V1 failures this replaces (see ``TRAINING_HEALTH_TRACKER.md`` and
``todos/v2-monitoring.md``):

* ``patience_counter += 1`` was incremented on epochs where validation was
  *skipped*, comparing a stale cached metric that could never win. Nominal
  ``patience=8`` was worth as few as 4 real measurements. A history of
  validated epochs only, re-reduced each time, cannot express that bug.
* The stop signal was ``val_f1_macro`` at a frozen threshold 0.2653 against a
  0.7927 break-even — a ~14:1 calibration-noise-to-signal ratio. Selection is
  now ``val_mAP`` (upstream, ``_SELECTION_METRICS`` in ``train_direct.py``).
* The operative rule was in fact a human reading canary #1 ("require >=1.01x
  mAP growth per epoch"), which flagged a monotonically-improving run as
  unhealthy from its second steady-state epoch. Per-epoch growth ratios are
  not a usable statistic at mAP > 0.65; :func:`plateau` compares the best of
  one window against the best of the previous window instead.

Rule set:

``plateau``        best(last k) - best(previous k) < ``min_delta``, confirmed
                   over ``confirm`` consecutive evaluations. The primary
                   "training has converged" signal.
``peak_decline``   the selection metric has sat ``regression_delta`` below its
                   all-time best for ``regression_confirm`` consecutive
                   validated epochs. The real "peak and decline" signal
                   (v2-plan.md SS9.1 item 3).
``overfit_loss``   ``val_loss`` rising while ``train_loss`` falls, for
                   ``overfit_confirm`` consecutive validated epochs. Canary #6
                   as code — the only direct read on the thing V1 was trying to
                   avoid, and it never fired at either V1 stop.
``budget``         informational: where the run sits against its epoch budget.

Advisory-only (never contributes to the action, per ``v2-monitoring.md``'s
"none of these are sufficient cause to stop training on their own"):
missing-positive / memorization fingerprint drift, via
:func:`fingerprint_advisories`.

``min_delta`` defaults to **3e-3**, the val instrument's resolution floor from
v2-plan.md SS8.3 (200-bin AP bias -0.0031; 30K sampling CI +-0.0027; the config's
own "treat val_mAP differences below ~0.003 as noise"). That floor *straddles* a
single epoch's real gain -- V1-P2's per-epoch mAP gains were +0.0041, +0.0045,
+0.0015, +0.0031, +0.0022, two of five below it -- which is why the plateau test
is window-over-window rather than per-epoch. Per-epoch, that healthy curve scores
non-improvements often enough to trip a patience of 2; the best of a 3-epoch
window against the best of the previous three gives +0.0067.

**Know what 3e-3 is and is not.** Both of its components are near-constant WITHIN
a run: the val split is cached and deterministic, the val loader runs
``shuffle=False`` and ``random_flip_prob=0``, so the val pass is a deterministic
function of the weights and the same 30K images are scored every epoch. The
sampling CI is therefore a fixed offset shared by every epoch, and the plan
records the binning bias as "largely cancel[ing] in checkpoint-to-checkpoint
comparison". So 3e-3 is the right floor for comparing ACROSS runs or configs, and
is *conservative* for comparing epochs within one run -- where the residual noise
is only the error filter dropping different samples (logged as a warning) and the
binning bias's small response to calibration shifts.

That makes this rule deliberately **slack**: it errs toward running too long, not
stopping too early, which is the correct direction of error for this project and
the reason the number is not being lowered on theory. V1-P2's five consecutive
same-sign mAP deltas with no reversal put within-run noise well under 0.0015.
**Re-derive it from V2's own first ~10 validated epochs** (residuals about a
smoothed trend) before tightening; do not lower it from an argument.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

__all__ = [
    "EVAL_RECORD_KEYS",
    "FINGERPRINT_DIRECTIONS",
    "StopPolicy",
    "Trigger",
    "StopVerdict",
    "make_record",
    "append_record",
    "phase_history",
    "evaluate",
    "fingerprint_advisories",
]


# Scalars carried per validated epoch. Anything a rule reads must live here, so
# that the rule stays a pure function of the checkpointed history.
EVAL_RECORD_KEYS = (
    "epoch",            # 1-based, phase-local (matches training_state.epoch)
    "global_step",
    "selection_value",  # the scalar driving selection, per config.training.selection_metric
    "val_mAP",
    "val_loss",
    "train_loss",
    "val_f1_macro",
    "val_f1_micro",
    "lr",
    "loss_id",           # identifies the loss FUNCTION (ASL gamma_neg); see overfit_loss
    "clean_margin",      # sibling-margin diagnostic; see CLEAN_MARGIN_NOTE
)

# ``asl_val/sibling_gap_macro`` from asl_telemetry: for val rows where exactly one
# member of a confusable group (hair_color, eye_color, hair_length, breast_size) is
# labelled positive, the mean of
#     p(labelled sibling) - max p(unlabelled siblings)
# A falling margin may corroborate an existing decline. Holding/rising margins
# cannot distinguish learning from memorized wrong positives, so never demote
# a stop trigger. This signal cannot stop training on its own (V2 section 9.2).
CLEAN_MARGIN_NOTE = "asl_val/sibling_gap_macro"

# Direction that indicates a *problem*, for the advisory fingerprints
# (v2-monitoring.md "Missing-positive bias diagnostics"). "down" = falling is
# the problem direction.
#
# CRITICAL: all three of these are calibration-coupled, and LR decay tightens
# calibration in the problem direction as a matter of course. canary #7 says so
# explicitly about the closely-related mean_active: "Healthy: monotonic decline
# as logits tighten with LR decay." So a bare "two consecutive moves down"
# test fires on every healthy run -- which is canary #1's failure mode rebuilt
# one layer down, and the way an operator learns to ignore a flag. Hence the
# stall gate in fingerprint_advisories(): a fingerprint only flags when the
# selection metric is NOT clearly improving over the same span, which is what
# v2-monitoring.md actually specifies ("confirmatory diagnostics for a val/mAP
# pattern that already looks suspect").
FINGERPRINT_DIRECTIONS = {
    "pred_pos_ratio": "down",
    "mean_sigmoid_topK_unlabeled": "down",
    "logit_std_rank_11_50": "down",
}

# Absolute levels below which a fingerprint is meaningful at all. Cole's EPR
# result is about pred_pos_ratio falling *below ~1.0* (predicting fewer positives
# than are labelled = suppression collapse), not about it falling: a model
# emitting 3x the labelled positives and tightening toward 1.5x is calibrating,
# not collapsing. Encoding that row as direction-only was wrong.
#
# NOTE this level is only interpretable when inference.prediction_threshold is
# actually calibrated for the CURRENT model. At V1's stale 0.2653 the ratio was
# ~128 and the test would never fire; at the fitted 0.7927 it lands near 1.
FINGERPRINT_LEVEL_FLOORS = {
    "pred_pos_ratio": 1.0,
}


# --------------------------------------------------------------------------
# Policy
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class StopPolicy:
    """Immutable snapshot of the stop configuration for one evaluation.

    Built from ``config.training`` by :meth:`from_config` so the rules never
    reach into a live config object (and so tests can construct one directly).
    """

    mode: str = "advise"          # 'advise' | 'halt' | 'off'
    min_delta: float = 3e-3       # plateau: required window-over-window gain
    window: int = 3               # plateau: k
    confirm: int = 2              # plateau: consecutive confirmations
    min_epochs: int = 4           # no rule may fire before this many validated epochs
    burn_in_epochs: int = 0       # epochs excluded from the stop history entirely
    regression_delta: float = 6e-3
    regression_confirm: int = 2
    overfit_confirm: int = 3
    max_epochs: int = 0           # the hard cap; 0 = unknown/unbounded

    @classmethod
    def from_config(cls, training_cfg: Any) -> "StopPolicy":
        """Build the policy, applying the per-phase override for training.phase.

        The plateau rule cannot confirm before ``burn_in + 2*window + confirm - 1``
        validated epochs. On a long from-scratch phase that is cheap; on a short
        fine-tune it can consume most of the budget, and on a 3-4 epoch detail
        phase it can never be reached -- structurally the same silent no-op as the
        LR gate this system replaced. So the geometry is per-phase, keyed the same
        way ``asl_schedule`` is.
        """
        def _get(name: str, default):
            value = getattr(training_cfg, name, default)
            return default if value is None else value

        # Phase override first, so the explicit per-phase value wins over the
        # top-level default for every key it sets.
        phase = int(_get("phase", 1))
        overrides: Dict[str, Any] = {}
        phases_cfg = getattr(training_cfg, "early_stopping_phases", None)
        phase_cfg = getattr(phases_cfg, f"phase{phase}", None) if phases_cfg else None
        if phase_cfg is not None:
            for key in ("mode", "window", "confirm", "min_epochs"):
                value = getattr(phase_cfg, key, None)
                if value is not None:
                    overrides[key] = value

        def _resolve(policy_key: str, config_key: str, default, cast):
            if policy_key in overrides:
                return cast(overrides[policy_key])
            return cast(_get(config_key, default))

        return cls(
            mode=str(_resolve("mode", "early_stopping_mode", "advise", str)).lower(),
            window=_resolve("window", "early_stopping_window", 3, int),
            confirm=_resolve("confirm", "early_stopping_confirm", 2, int),
            min_epochs=_resolve("min_epochs", "early_stopping_min_epochs", 4, int),
            min_delta=float(_get("early_stopping_min_delta", 3e-3)),
            burn_in_epochs=int(_get("early_stopping_burn_in_epochs", 0)),
            regression_delta=float(_get("early_stopping_regression_delta", 6e-3)),
            regression_confirm=int(_get("early_stopping_regression_confirm", 2)),
            overfit_confirm=int(_get("early_stopping_overfit_confirm", 3)),
            max_epochs=int(_get("num_epochs", 0)),
        )

    def first_confirmable_epoch(self) -> int:
        """Earliest validated epoch at which any rule can confirm."""
        return max(
            self.burn_in_epochs + 2 * self.window + self.confirm - 1,
            self.burn_in_epochs + self.min_epochs,
        )

    def describe(self) -> str:
        return (
            f"mode={self.mode} plateau(min_delta={self.min_delta:g}, window={self.window}, "
            f"confirm={self.confirm}) peak_decline(delta={self.regression_delta:g}, "
            f"confirm={self.regression_confirm}) overfit(confirm={self.overfit_confirm}) "
            f"min_epochs={self.min_epochs} burn_in={self.burn_in_epochs} "
            f"budget={self.max_epochs or 'unbounded'}"
        )


# --------------------------------------------------------------------------
# Verdict types
# --------------------------------------------------------------------------

@dataclass
class Trigger:
    """One rule's outcome.

    ``state`` is deliberately three-valued. ``pending`` (not enough history, or
    inside the min-epochs floor) must never be collapsed into ``ok``: the old
    code's habit of treating "no measurement" as "no improvement" is exactly how
    patience accrued on epochs that measured nothing.
    """

    name: str
    state: str                      # 'tripped' | 'ok' | 'pending'
    detail: str = ""
    values: Dict[str, float] = field(default_factory=dict)
    advisory: bool = False          # advisory triggers never affect the action

    @property
    def tripped(self) -> bool:
        return self.state == "tripped"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class StopVerdict:
    action: str                     # 'continue' | 'advise' | 'halt'
    triggers: List[Trigger]
    n_validations: int
    best_value: Optional[float]
    best_epoch: Optional[int]
    epochs_since_best: Optional[int]
    policy: StopPolicy

    @property
    def stop_triggers(self) -> List[Trigger]:
        return [t for t in self.triggers if t.tripped and not t.advisory]

    @property
    def advisories(self) -> List[Trigger]:
        return [t for t in self.triggers if t.tripped and t.advisory]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action": self.action,
            "n_validations": self.n_validations,
            "best_value": self.best_value,
            "best_epoch": self.best_epoch,
            "epochs_since_best": self.epochs_since_best,
            "policy": asdict(self.policy),
            "triggers": [t.to_dict() for t in self.triggers],
        }

    def summary_line(self) -> str:
        """One line for the training log. Always states the budget position."""
        parts = []
        for t in self.triggers:
            mark = {"tripped": "TRIP", "ok": "ok", "pending": "--"}[t.state]
            parts.append(f"{t.name}={mark}")
        best = "n/a" if self.best_value is None else f"{self.best_value:.6f}@E{self.best_epoch}"
        return (
            f"stop-check[{self.action.upper()}] n_val={self.n_validations} "
            f"best={best} since_best={self.epochs_since_best} " + " ".join(parts)
        )

    def reasons(self) -> str:
        """Human-readable justification for a non-continue action."""
        rows = [f"  - {t.name}: {t.detail}" for t in self.stop_triggers]
        return "\n".join(rows)


# --------------------------------------------------------------------------
# History maintenance
# --------------------------------------------------------------------------

def make_record(
    *,
    epoch: int,
    global_step: int,
    selection_value: float,
    val_mAP: float,
    val_loss: float,
    train_loss: float,
    val_f1_macro: float,
    val_f1_micro: float,
    lr: float,
    loss_id: Optional[Any] = None,
    clean_margin: Optional[float] = None,
    fingerprints: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """Build one history record. Plain dict on purpose.

    ``TrainingState`` is serialized with ``dataclasses.asdict`` and rebuilt with
    ``from_dict``; a list of nested dataclasses would round-trip to dicts and
    then fail to reconstruct. Plain dicts of floats are the only shape that
    survives that path unchanged, and they are also what the JSONL decision log
    wants.
    """
    record: Dict[str, Any] = {
        "epoch": int(epoch),
        "global_step": int(global_step),
        "selection_value": float(selection_value),
        "val_mAP": float(val_mAP),
        "val_loss": float(val_loss),
        "train_loss": float(train_loss),
        "val_f1_macro": float(val_f1_macro),
        "val_f1_micro": float(val_f1_micro),
        "lr": float(lr),
        "loss_id": loss_id,
        "clean_margin": (
            None if clean_margin is None or not math.isfinite(float(clean_margin))
            else float(clean_margin)
        ),
    }
    if fingerprints:
        record["fingerprints"] = {str(k): float(v) for k, v in fingerprints.items()}
    return record


def append_record(history: Sequence[Dict[str, Any]], record: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Append (or replace) ``record`` and return a new, epoch-ordered history.

    Replace-on-same-epoch is the soft-stop guarantee. A soft stop taken
    mid-epoch N rewinds to the start of epoch N; the resumed process re-runs and
    re-validates that epoch, and a blind append would then hold epoch N twice —
    inflating ``n_validations`` and letting a duplicated value satisfy a window
    on its own. Keyed replacement makes the history idempotent under replay.
    """
    out = [dict(r) for r in history if int(r.get("epoch", -1)) != int(record["epoch"])]
    out.append(dict(record))
    out.sort(key=lambda r: int(r.get("epoch", 0)))
    return out


def phase_history(
    history: Sequence[Dict[str, Any]], burn_in_epochs: int = 0
) -> List[Dict[str, Any]]:
    """Records eligible for stop decisions: past burn-in, and numerically sane.

    A non-finite ``selection_value`` is dropped rather than propagated. One NaN
    inside a window would poison every comparison that window takes part in
    (``max`` with a NaN is order-dependent), and a NaN cannot be an improvement
    or a decline — it is an absent measurement.
    """
    out = []
    for r in history:
        if int(r.get("epoch", 0)) <= int(burn_in_epochs):
            continue
        value = r.get("selection_value")
        if value is None or not math.isfinite(float(value)):
            continue
        out.append(dict(r))
    return out


# --------------------------------------------------------------------------
# Rules
# --------------------------------------------------------------------------

def _window_gain(values: Sequence[float], end: int, window: int) -> Optional[float]:
    """best(values[end-window:end]) - best(values[end-2*window:end-window]).

    ``end`` is exclusive, so this can be evaluated on any prefix of the history
    — which is how "confirmed over N consecutive evaluations" is derived without
    storing a counter. Returns None when the prefix is too short.
    """
    if window < 1 or end < 2 * window:
        return None
    recent = max(values[end - window:end])
    prior = max(values[end - 2 * window:end - window])
    return recent - prior


def plateau(values: Sequence[float], policy: StopPolicy) -> Trigger:
    """Window-over-window improvement below ``min_delta``, confirmed."""
    n = len(values)
    gains: List[Optional[float]] = [
        _window_gain(values, n - offset, policy.window) for offset in range(policy.confirm)
    ]
    latest = gains[0]
    if any(g is None for g in gains):
        need = 2 * policy.window + policy.confirm - 1
        return Trigger(
            name="plateau",
            state="pending",
            detail=(
                f"needs {need} validated epochs past burn-in to confirm "
                f"(window={policy.window} x2 + {policy.confirm - 1} prior evaluations); have {n}"
            ),
            values={"latest_gain": float("nan") if latest is None else latest},
        )

    checked = [float(g) for g in gains]  # type: ignore[arg-type]
    if all(g < policy.min_delta for g in checked):
        return Trigger(
            name="plateau",
            state="tripped",
            detail=(
                f"best(last {policy.window}) - best(prior {policy.window}) = {checked[0]:+.6f} "
                f"< min_delta {policy.min_delta:g}, for {policy.confirm} consecutive evaluations "
                f"(gains {', '.join(f'{g:+.6f}' for g in checked)})"
            ),
            values={"latest_gain": checked[0], "min_delta": policy.min_delta},
        )
    return Trigger(
        name="plateau",
        state="ok",
        detail=(
            f"window gain {checked[0]:+.6f} vs min_delta {policy.min_delta:g}"
            + ("" if len(checked) == 1 else f" (prior evaluations {', '.join(f'{g:+.6f}' for g in checked[1:])})")
        ),
        values={"latest_gain": checked[0], "min_delta": policy.min_delta},
    )


def peak_decline(values: Sequence[float], policy: StopPolicy) -> Trigger:
    """Sustained decline from the all-time best.

    Kept separate from ``plateau`` because a flat curve and a falling curve call
    for different responses, and collapsing them into one patience counter is
    what made V1's stop rule uninformative.

    **This rule cannot tell you WHY the metric is falling, and at this project's
    label-noise levels the benign cause is the more likely one.** v2-plan.md SS9.1:
    measured AP is biased *downward*, and the bias *grows with the model's true
    quality*, because a model that has learned a concept ranks its
    truly-positive-but-labelled-negative val images high and AP punishes a
    false positive at rank 1 far more than at rank 500. Measured missing rates
    are ~0.3-0.5% at the head, ~5% median in the tail decile and 23-64% on
    individual confusable tags. So a decline here is equally consistent with
    "the model got worse" and "the model outgrew the annotation" -- and in the
    second case stopping is exactly the wrong response.

    Treat a trip as an instruction to disambiguate, not as a verdict. The only
    thing that can disambiguate it is a low-noise metric: SS9.2's sibling-negative
    evaluation, or the gold slice. Neither is wired into this loop yet, which is
    the strongest reason ``early_stopping_mode`` defaults to ``advise``.
    """
    n = len(values)
    c = max(1, policy.regression_confirm)
    if n < c + 1:
        return Trigger(
            name="peak_decline",
            state="pending",
            detail=f"needs {c + 1} validated epochs past burn-in; have {n}",
        )
    best = max(values)
    tail = list(values[-c:])
    floor = best - policy.regression_delta
    if all(v < floor for v in tail):
        return Trigger(
            name="peak_decline",
            state="tripped",
            detail=(
                f"last {c} validated epochs ({', '.join(f'{v:.6f}' for v in tail)}) all sit more "
                f"than {policy.regression_delta:g} below the best {best:.6f}"
            ),
            values={"best": best, "last": tail[-1], "gap": best - tail[-1]},
        )
    return Trigger(
        name="peak_decline",
        state="ok",
        detail=f"last {tail[-1]:.6f} vs best {best:.6f} (gap {best - tail[-1]:+.6f})",
        values={"best": best, "last": tail[-1], "gap": best - tail[-1]},
    )


def overfit_loss(
    val_losses: Sequence[float],
    train_losses: Sequence[float],
    policy: StopPolicy,
    loss_ids: Optional[Sequence[Any]] = None,
) -> Trigger:
    """Canary #6 red: val_loss rising for c consecutive epochs while train_loss falls.

    Requires ``c`` consecutive *increases*, hence ``c + 1`` points. Both
    directions must hold: val_loss rising while train_loss also rises is an
    instability or a data incident, not overfitting, and should not be reported
    as the latter.

    ``loss_ids`` identifies the loss FUNCTION each point was measured under (in
    practice the ASL ``gamma_neg``). A ``gamma_neg`` step changes what the number
    means, so a window that straddles one is comparing two different objectives
    and cannot support a monotonicity claim -- it returns ``pending``. Because
    gamma steps are manual (stop -> set override -> resume), such a window is
    exactly what a resumed run would otherwise evaluate first.

    **Read the SCOPE of this rule narrowly.** It detects classic overfitting:
    fitting train-set idiosyncrasy that the val set does not share. It is
    structurally blind to this project's actual concern, missing-positive noise
    memorization, because train and val are drawn from the SAME noisy annotation
    process -- memorizing that shared noise *lowers* val_loss. See
    ``FINGERPRINT_DIRECTIONS`` for the diagnostics that do look at unlabelled
    behaviour, and Zhao & Gomes (arXiv:2102.08427) for why a noisy-val loss
    cannot arbitrate this.
    """
    c = max(1, policy.overfit_confirm)
    span = c + 1
    if len(val_losses) < span or len(train_losses) < span:
        return Trigger(
            name="overfit_loss",
            state="pending",
            detail=f"needs {span} validated epochs past burn-in; have {min(len(val_losses), len(train_losses))}",
        )
    vl = [float(v) for v in val_losses[-span:]]
    tl = [float(v) for v in train_losses[-span:]]
    if not all(math.isfinite(v) for v in vl + tl):
        return Trigger(name="overfit_loss", state="pending", detail="non-finite loss in window")
    if loss_ids is not None and len(loss_ids) >= span:
        window_ids = list(loss_ids[-span:])
        distinct = {repr(i) for i in window_ids}
        if len(distinct) > 1:
            return Trigger(
                name="overfit_loss",
                state="pending",
                detail=(
                    f"window straddles a loss-shape change ({' -> '.join(str(i) for i in window_ids)}); "
                    f"val_loss is not comparable across it"
                ),
            )
    val_rising = all(vl[i] > vl[i - 1] for i in range(1, span))
    train_falling = all(tl[i] < tl[i - 1] for i in range(1, span))
    if val_rising and train_falling:
        return Trigger(
            name="overfit_loss",
            state="tripped",
            detail=(
                f"val_loss rose for {c} consecutive validated epochs "
                f"({' -> '.join(f'{v:.6g}' for v in vl)}) while train_loss fell "
                f"({' -> '.join(f'{v:.6g}' for v in tl)})"
            ),
            values={"val_loss": vl[-1], "train_loss": tl[-1]},
        )
    return Trigger(
        name="overfit_loss",
        state="ok",
        detail=(
            f"val_loss {'rising' if val_rising else 'not rising'}, "
            f"train_loss {'falling' if train_falling else 'not falling'} over the last {c} steps"
        ),
        values={"val_loss": vl[-1], "train_loss": tl[-1]},
    )


def clean_margin(records: Sequence[Dict[str, Any]], span: int) -> Trigger:
    """Report the trend of the sibling-margin diagnostic over the last ``span`` records.

    The window is ALIGNED to the record tail, not to the filtered series: the
    last ``span`` records are taken first, and only the margins carried by
    THOSE records are read. If any of them lack a margin (ASL telemetry
    hiccup, sibling_groups_path failing to resolve on a given resume), the
    trigger is ``pending`` rather than silently measuring an OLDER span.
    Missing values leave the diagnostic pending; stale observations cannot
    corroborate a current decline.
    """
    # Aligned tail: the last `span` records, holes and all.
    tail_records = list(records)[-max(1, span):]
    series = [
        float(r["clean_margin"]) for r in tail_records
        if r.get("clean_margin") is not None and math.isfinite(float(r["clean_margin"]))
    ]
    if len(series) < len(tail_records) or len(series) < 2:
        missing = len(tail_records) - len(series)
        return Trigger(
            name="clean_margin",
            state="pending",
            detail=(
                f"needs {span} consecutive validated epochs carrying {CLEAN_MARGIN_NOTE} "
                f"in the aligned tail; have {len(series)} of {len(tail_records)} "
                f"({missing} missing). Cannot corroborate a decline with incomplete observations: "
                f"the diagnostic requires the current window. "
                f"Check that training.asl_telemetry is enabled and sibling_groups_path resolves."
            ),
        )
    change = series[-1] - series[0]
    return Trigger(
        name="clean_margin",
        state="ok",
        detail=(
            f"{CLEAN_MARGIN_NOTE} {series[0]:.6f} -> {series[-1]:.6f} ({change:+.6f}) over the "
            f"last {len(series)} validated epochs"
        ),
        values={"latest": series[-1], "change": change, "n": float(len(series))},
    )


def budget(completed_epochs: int, policy: StopPolicy) -> Trigger:
    """Informational: position against the stated epoch budget.

    Never trips. The budget is enforced by the epoch loop itself; this exists so
    every stop-check line in the log states where the run sits against the
    number that was committed up front ("state it up front and hold to it",
    v2-monitoring.md decision rules).
    """
    if policy.max_epochs <= 0:
        return Trigger(name="budget", state="pending", detail="no epoch budget configured")
    remaining = policy.max_epochs - int(completed_epochs)
    return Trigger(
        name="budget",
        state="ok",
        detail=f"epoch {int(completed_epochs)}/{policy.max_epochs} ({remaining} remaining)",
        values={"completed": float(completed_epochs), "budget": float(policy.max_epochs),
                "remaining": float(remaining)},
    )


def fingerprint_advisories(
    history: Sequence[Dict[str, Any]],
    consecutive: int = 2,
    stall_delta: float = 0.0,
) -> List[Trigger]:
    """Advisory-only drift flags for the missing-positive fingerprints.

    ``v2-monitoring.md``: "Two consecutive monotonic moves in the 'problem'
    direction is the trigger to (a) spot-check specific predictions early ...
    None are sufficient cause to stop training on their own" -- and, crucially,
    they are "confirmatory diagnostics for a ``val/mAP`` pattern that already
    looks suspect."

    That last clause is a gate, not decoration. All three fingerprints are
    calibration-coupled, and LR decay moves all three in the problem direction
    during entirely healthy training. Flagging on direction alone would fire
    every epoch of a good run. So a fingerprint flags only when the selection
    metric failed to improve by more than ``stall_delta`` across the same span.
    """
    out: List[Trigger] = []
    span = consecutive + 1
    for name, bad_direction in sorted(FINGERPRINT_DIRECTIONS.items()):
        # Keep the fingerprint and the selection metric on the SAME records, so
        # the stall gate is measured over exactly the span being flagged.
        paired = [
            (float(r["fingerprints"][name]), float(r["selection_value"]))
            for r in history
            if isinstance(r.get("fingerprints"), dict) and name in r["fingerprints"]
            and math.isfinite(float(r["fingerprints"][name]))
        ]
        if len(paired) < span:
            continue
        tail = [fp for fp, _ in paired[-span:]]
        sel = [sv for _, sv in paired[-span:]]
        if bad_direction == "down":
            moved = all(tail[i] < tail[i - 1] for i in range(1, span))
        else:
            moved = all(tail[i] > tail[i - 1] for i in range(1, span))
        if not moved:
            continue

        # Level gate (Cole's EPR is about crossing ~1.0, not about falling).
        floor = FINGERPRINT_LEVEL_FLOORS.get(name)
        if floor is not None and tail[-1] >= floor:
            continue

        # Stall gate: if the selection metric clearly improved across the same
        # span, the fingerprint's drift is calibration tightening, not
        # suppression, and there is no suspect pattern to confirm.
        sel_change = sel[-1] - sel[0]
        if sel_change > stall_delta:
            continue

        out.append(
            Trigger(
                name=f"fingerprint:{name}",
                state="tripped",
                advisory=True,
                detail=(
                    f"{consecutive} consecutive moves {bad_direction} "
                    f"({' -> '.join(f'{v:.6g}' for v in tail)}) WHILE the selection metric "
                    f"failed to improve over the same span ({sel_change:+.6f} <= "
                    f"{stall_delta:g})"
                    + (f" and the level is below {floor:g}" if floor is not None else "")
                    + " — confirmatory diagnostic only, not a stop cause"
                ),
                values={
                    "latest": tail[-1],
                    "delta": tail[-1] - tail[0],
                    "selection_change": sel_change,
                },
            )
        )
    return out


# --------------------------------------------------------------------------
# Top level
# --------------------------------------------------------------------------

def _best(values: Sequence[float]) -> Tuple[Optional[int], Optional[float]]:
    if not values:
        return None, None
    idx = max(range(len(values)), key=lambda i: values[i])
    return idx, values[idx]


def evaluate(
    history: Sequence[Dict[str, Any]],
    policy: StopPolicy,
    completed_epochs: int = 0,
) -> StopVerdict:
    """Reduce the whole persisted history to a single verdict.

    Pure: same history + same policy => same verdict, in this process or in one
    resumed from a checkpoint an hour later.
    """
    records = phase_history(history, policy.burn_in_epochs)
    values = [float(r["selection_value"]) for r in records]
    val_losses = [float(r.get("val_loss", float("nan"))) for r in records]
    train_losses = [float(r.get("train_loss", float("nan"))) for r in records]
    n = len(values)

    best_idx, best_value = _best(values)
    best_epoch = int(records[best_idx]["epoch"]) if best_idx is not None else None
    epochs_since_best = (n - 1 - best_idx) if best_idx is not None else None

    triggers: List[Trigger] = []
    below_floor = n < max(1, policy.min_epochs)
    if below_floor:
        # Explicit floor replaces V1's accidental one. The old code gated
        # patience on `lr / scheduler.max_lr < 0.5`, which on a single cosine
        # cycle is false until the midpoint of the anneal — so the auto-stop
        # could not fire before ~epoch 8.5 of P2's 15, and with patience=4 the
        # earliest possible stop was ~epoch 13. A floor should be a stated
        # number of epochs, not a side effect of the LR schedule.
        reason = f"below min_epochs floor ({n} < {policy.min_epochs})"
        triggers.extend([
            Trigger(name="plateau", state="pending", detail=reason),
            Trigger(name="peak_decline", state="pending", detail=reason),
            Trigger(name="overfit_loss", state="pending", detail=reason),
        ])
    else:
        triggers.append(plateau(values, policy))
        triggers.append(peak_decline(values, policy))
        triggers.append(overfit_loss(
            val_losses, train_losses, policy,
            loss_ids=[r.get("loss_id") for r in records],
        ))

    triggers.append(budget(completed_epochs, policy))

    # Low-noise arbiter. Reported every epoch; consulted below.
    _clean_span = max(2, policy.regression_confirm + 1)
    _clean = clean_margin(records, _clean_span)
    triggers.append(_clean)

    # Sibling margin is veto-only: rising/holding values are uninformative
    # about memorized wrong positives and cannot weaken a decline trigger.
    _decline = next((t for t in triggers if t.name == "peak_decline"), None)
    if _decline is not None and _decline.tripped and _clean.state == "ok":
        _change = _clean.values.get("change", 0.0)
        if _change < 0.0:
            _decline.detail += (
                f" -- CORROBORATED: sibling margin fell over the same span "
                f"({CLEAN_MARGIN_NOTE} {_change:+.6f})."
            )

    # stall_delta = min_delta: a fingerprint only confirms a pattern that the
    # selection metric already fails to contradict.
    triggers.extend(fingerprint_advisories(records, stall_delta=policy.min_delta))

    fired = [t for t in triggers if t.tripped and not t.advisory]
    if policy.mode == "off" or not fired:
        action = "continue"
    elif policy.mode == "halt":
        action = "halt"
    else:
        action = "advise"

    return StopVerdict(
        action=action,
        triggers=triggers,
        n_validations=n,
        best_value=best_value,
        best_epoch=best_epoch,
        epochs_since_best=epochs_since_best,
        policy=policy,
    )
