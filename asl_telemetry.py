"""ASL gamma_neg drive machinery + always-on telemetry.

Implements the manual-fallback minimum of todos/ASL_plan.md (SS3, SS5, SS8):

- Guarded manual gamma_neg steps (phase clamps, hold windows, min dwell) --
  the operative driver per the 2026-07-02 adversarial review. Steps are
  requested by editing ``training.tag_loss.gamma_neg_override`` in the YAML
  and restarting (stop / edit / resume); this module validates and applies
  them at startup.
- Loss-state persistence: the manager owns a plain-dict ``state`` that the
  trainer attaches to ``TrainingState.loss_state``, so every checkpoint save
  carries gamma_neg + telemetry EMAs and a resume restores them, OVERRIDING
  the YAML value (SS8 row 2 -- without this, any gamma change silently
  reverts to YAML on restart).
- Always-on telemetry (SS5): dp_mean, dp_hard (top-K non-GT gap),
  threshold-free per-decile EPR (Cole 2021 formulation), non-GT score
  histogram with the [0.2, 0.5] clip watch band, and per-confusable-group
  sibling-gap. Train-side samples ride the already-computed detached logits
  at optimizer-update boundaries; val-side variants consume the accumulated
  probability/target matrices.
- Shadow controller (SS4, demoted 2026-07-02): logs the gamma the paper's
  adaptive-asymmetry law WOULD set next to the actual gamma. Zero authority.

Measurement hygiene (SS5): all metrics are computed on columns >= 2 (PAD=0
and UNK=1 are live, loss-free, drifting outputs), probabilities are fp32
upcast before sigmoid, and rating tags are excluded from EVERY column-space
metric -- dp_mean, dp_hard, the non-GT histogram AND the per-decile EPR.
Every row carries exactly one rating positive, so leaving them in inflates
mean(p_pos) and swamps the EPR denominator of whichever decile they sort into.

The golden/Anima set is deliberately NOT wired here: it is evaluation-only,
via the standalone tools/asl_anima_canary.py script.
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

logger = logging.getLogger(__name__)

# PAD=0, UNK=1 -- consistent with loss ignore_indices and val skip_metric_cols
SKIP_COLS = 2

RATING_PREFIX = "rating:"


def _window_for_phase(schedule_cfg, phase: int):
    """Return the ASLPhaseWindowConfig for a 1-based phase, or None."""
    if schedule_cfg is None:
        return None
    return getattr(schedule_cfg, f"phase{int(phase)}", None)


class ASLDriveManager:
    """Owns mutable gamma_neg + its persisted state + the SS5 telemetry set.

    The ``state`` dict is shared BY REFERENCE with TrainingState.loss_state:
    every mutation here lands in the next checkpoint automatically. Only
    JSON/pickle-friendly builtins (float/int/str/list/dict/None) may be
    stored in it -- asdict() snapshots it at each save.
    """

    def __init__(
        self,
        config,
        criterion,
        vocab,
        device: torch.device,
        state: Optional[Dict[str, Any]],
        start_epoch: int,
        monitor=None,
    ):
        self.criterion = criterion
        self.monitor = monitor
        self.device = device
        self.state: Dict[str, Any] = state if isinstance(state, dict) else {}

        training_cfg = config.training
        self.phase = int(getattr(training_cfg, "phase", 0) or 0)
        self.sched = getattr(training_cfg, "asl_schedule", None)
        self.tele = getattr(training_cfg, "asl_telemetry", None)
        self.window = _window_for_phase(self.sched, self.phase)

        self._reconcile_gamma(training_cfg.tag_loss, start_epoch)
        self._init_telemetry(vocab)

    # ------------------------------------------------------------------
    # gamma_neg reconciliation + guarded manual steps (SS3, SS8)
    # ------------------------------------------------------------------

    # State keys whose meaning is PHASE-LOCAL: epoch counters reset to 0 at a
    # phase transition (train_direct.py), so a value inherited from the previous
    # phase is not just stale, it is in a different coordinate system. Carrying
    # `gamma_last_change_epoch` forward makes the dwell guard compute a NEGATIVE
    # elapsed and refuse every step for `last_change + 3` epochs of the new
    # phase; carrying `epr_baseline*` compares 448px EPR against a 320px
    # baseline. `gamma_neg` and `gamma_history` are deliberately NOT in this set:
    # gamma carries over frozen across a transition (ASL_plan SS3).
    _PHASE_LOCAL_STATE_KEYS = (
        "gamma_last_change_epoch",
        "epr_baseline",
        "epr_baseline_epoch",
        "epr_baseline_pending_epoch",
        "telemetry",
    )

    def _reconcile_gamma(self, tag_loss_cfg, start_epoch: int) -> None:
        # Must run BEFORE self.state["phase"] is overwritten below.
        persisted_phase = self.state.get("phase")
        # `self.phase > 0` mirrors train_direct's phase_transition predicate
        # exactly. Without it, running with training.phase unset (0) against a
        # phase-2 checkpoint would drop the dwell bookkeeping even though the
        # trainer did NOT reset the epoch counters, letting a manual gamma step
        # land earlier than min_dwell_epochs allows.
        if (persisted_phase is not None and self.phase > 0
                and int(persisted_phase) != self.phase):
            dropped = [k for k in self._PHASE_LOCAL_STATE_KEYS if k in self.state]
            for key in dropped:
                self.state.pop(key, None)
            logger.warning(
                "ASL drive: PHASE CHANGE %s -> %d detected in the persisted loss state. "
                "Dropped phase-local keys %s (epoch counters are phase-local; keeping "
                "them would make the dwell guard refuse every gamma step for the first "
                "~%d epochs of the new phase and compare EPR against the previous "
                "phase's baseline). gamma_neg and gamma_history carry over.",
                persisted_phase, self.phase, dropped or "<none>",
                int(getattr(self.sched, "min_dwell_epochs", 3)),
            )

        yaml_gamma = float(tag_loss_cfg.gamma_neg)
        persisted = self.state.get("gamma_neg")

        if persisted is None:
            self.gamma = yaml_gamma
            logger.info(
                "ASL drive: gamma_neg=%.3f from YAML (no persisted loss state: "
                "fresh run, or a checkpoint predating loss-state persistence).",
                self.gamma,
            )
        else:
            self.gamma = float(persisted)
            if abs(self.gamma - yaml_gamma) > 1e-9:
                logger.warning(
                    "ASL drive: gamma_neg=%.3f RESTORED from checkpoint loss state, "
                    "overriding YAML value %.3f (ASL_plan SS8: the checkpoint wins on "
                    "resume; use training.tag_loss.gamma_neg_override for a manual step).",
                    self.gamma,
                    yaml_gamma,
                )
            else:
                logger.info(
                    "ASL drive: gamma_neg=%.3f restored from checkpoint (matches YAML).",
                    self.gamma,
                )

        self.state["gamma_neg"] = float(self.gamma)
        self.state["phase"] = self.phase
        self.state.setdefault("gamma_history", [])

        override = getattr(tag_loss_cfg, "gamma_neg_override", None)
        if override is not None:
            override = float(override)
            if abs(override - self.gamma) <= 1e-9:
                logger.info(
                    "ASL drive: gamma_neg_override=%.3f equals current gamma_neg -- no-op "
                    "(clear the override in the YAML once the step has been taken).",
                    override,
                )
            else:
                self.request_gamma_step(override, start_epoch, source="yaml_override")

        if self.window is not None:
            lo, hi = float(self.window.gamma_neg_min), float(self.window.gamma_neg_max)
            if not (lo - 1e-9 <= self.gamma <= hi + 1e-9):
                logger.warning(
                    "ASL drive: gamma_neg=%.3f is OUTSIDE the phase %d window [%.1f, %.1f]. "
                    "Existing value is kept (windows constrain steps, not the inherited "
                    "value), but review todos/ASL_plan.md SS3.",
                    self.gamma, self.phase, lo, hi,
                )

        # Push the reconciled value into the live criterion (overrides whatever
        # the criterion was constructed with from YAML).
        self.criterion.set_gamma_neg(self.gamma)

    def request_gamma_step(self, target: float, epoch0: int, source: str = "manual") -> bool:
        """Apply a guarded gamma_neg step. Returns True if applied.

        epoch0 is the 0-based epoch about to run; guards use 1-based epochs
        (phase-local -- phase transitions reset epoch counters).
        """
        epoch1 = int(epoch0) + 1
        target = float(target)

        if self.sched is not None and getattr(self.sched, "enabled", True):
            w = self.window
            if w is not None:
                hold = int(getattr(w, "hold_epochs", 0))
                if epoch1 <= hold:
                    logger.error(
                        "ASL drive: REFUSING gamma_neg step %.3f -> %.3f at epoch %d: "
                        "phase %d holds gamma frozen through epoch %d (ASL_plan SS3 "
                        "hold/re-warmup window). Clear gamma_neg_override or wait.",
                        self.gamma, target, epoch1, self.phase, hold,
                    )
                    return False

            last = self.state.get("gamma_last_change_epoch")
            dwell = int(getattr(self.sched, "min_dwell_epochs", 3))
            if last is not None and (epoch1 - int(last)) < dwell:
                logger.error(
                    "ASL drive: REFUSING gamma_neg step %.3f -> %.3f at epoch %d: "
                    "last change was at epoch %d and min dwell is %d epochs "
                    "(ASL_plan SS3). Clear gamma_neg_override or wait.",
                    self.gamma, target, epoch1, int(last), dwell,
                )
                return False

            if w is not None:
                lo, hi = float(w.gamma_neg_min), float(w.gamma_neg_max)
                clamped = min(max(target, lo), hi)
                if abs(clamped - target) > 1e-9:
                    logger.warning(
                        "ASL drive: requested gamma_neg %.3f clamped to %.3f "
                        "(phase %d window [%.1f, %.1f]).",
                        target, clamped, self.phase, lo, hi,
                    )
                    target = clamped
                    if abs(target - self.gamma) <= 1e-9:
                        logger.error("ASL drive: step is a no-op after clamping; refused.")
                        return False

            if abs(target - self.gamma) > 1.0 + 1e-9:
                logger.warning(
                    "ASL drive: step |%.3f -> %.3f| exceeds 1 unit; ASL_plan SS3 "
                    "prescribes unit steps (7 -> 6 -> 5) with >=%d-epoch dwell.",
                    self.gamma, target, int(getattr(self.sched, "min_dwell_epochs", 3)),
                )

        old = self.gamma
        self.gamma = target
        self.criterion.set_gamma_neg(self.gamma)
        self.state["gamma_neg"] = float(self.gamma)
        self.state["gamma_last_change_epoch"] = epoch1
        self.state["gamma_history"] = list(self.state.get("gamma_history", [])) + [
            {"epoch": epoch1, "phase": self.phase, "from": float(old),
             "to": float(self.gamma), "source": source}
        ]
        # Snapshot the per-decile EPR at step time: the SS5 trend alarm compares
        # against this baseline for epr_alarm_window_epochs after the step.
        tele_state = self.state.get("telemetry") or {}
        epr = tele_state.get("epr_deciles")
        # A list of all-NaN is truthy but useless as a baseline -- require at
        # least one finite decile before accepting it.
        _usable = bool(epr) and any(
            isinstance(v, (int, float)) and math.isfinite(v) for v in epr
        )
        self.state["epr_baseline"] = list(epr) if _usable else None
        self.state["epr_baseline_epoch"] = epoch1
        # If no EPR has been logged yet there is nothing to snapshot -- which is
        # the normal case for a step taken at startup (_reconcile_gamma runs
        # before _init_telemetry) and ALWAYS the case for the first step after a
        # phase change, since the stale telemetry is dropped there. Without this
        # flag the baseline would stay None forever and the SS5 trend alarm --
        # the gate this step is supposed to be watched by -- would be silently
        # dead. _log_train captures the first available EPR as the baseline.
        if not self.state["epr_baseline"]:
            self.state["epr_baseline_pending_epoch"] = epoch1
            logger.info(
                "ASL drive: no EPR sample yet, deferring the trend-alarm baseline to "
                "the first telemetry log after this step (epoch %d). NOTE the deferred "
                "snapshot is measured AFTER the step, so it anchors the trend rather "
                "than capturing the step's own immediate effect.", epoch1,
            )
        else:
            # A valid baseline was captured here, so any marker left by an earlier
            # deferred step is stale; leaving it would let the next _log_train
            # overwrite this baseline and re-anchor it to the older epoch.
            self.state.pop("epr_baseline_pending_epoch", None)
        logger.warning(
            "ASL drive: gamma_neg STEP APPLIED %.3f -> %.3f at epoch %d (phase %d, "
            "source=%s). Gates to watch (ASL_plan SS5): per-decile EPR trend, "
            "dp_hard holds-or-widens, sibling-gap, Anima recall canary "
            "(tools/asl_anima_canary.py before/after the step).",
            old, self.gamma, epoch1, self.phase, source,
        )
        return True

    # ------------------------------------------------------------------
    # Telemetry (SS5)
    # ------------------------------------------------------------------

    def _init_telemetry(self, vocab) -> None:
        t = self.tele
        self.enabled = bool(t is not None and getattr(t, "enabled", False))
        if not self.enabled:
            logger.warning(
                "ASL telemetry DISABLED -- the SS5 always-on gate set (EPR trend, "
                "dp_hard, histogram, sibling-gap) will not be computed. Manual "
                "gamma steps without it degrade to flying blind (ASL_plan SS8)."
            )
            return

        self.interval = max(1, int(t.interval_updates))
        self.log_interval = max(self.interval, int(t.log_every_updates))
        self.beta = float(t.ema_beta)
        self.topk = int(t.topk_hard)
        self.num_deciles = int(t.num_deciles)
        self.hist_min = float(t.hist_min)
        self.hist_max = float(t.hist_max)
        self.hist_bins = int(t.hist_bins)
        self.band_low = float(t.watch_band_low)
        self.band_high = float(t.watch_band_high)
        self.epr_alarm_rel_drop = float(t.epr_alarm_rel_drop)
        self.epr_alarm_window = int(t.epr_alarm_window_epochs)
        self.shadow_enabled = bool(t.shadow_controller_enabled)
        self.shadow_lambda = float(t.shadow_lambda)
        self.shadow_target = float(t.shadow_delta_p_target)

        num_labels = len(vocab.tag_to_index)
        c_metric = num_labels - SKIP_COLS
        if c_metric <= 0:
            raise ValueError(f"Vocabulary too small for telemetry: {num_labels} labels")

        # --- Frequency deciles over metric columns (decile 0 = most frequent) ---
        freqs = torch.zeros(c_metric, dtype=torch.float64)
        tag_freqs = getattr(vocab, "tag_frequencies", {}) or {}
        for col in range(c_metric):
            tag = vocab.index_to_tag.get(col + SKIP_COLS)
            if tag is not None:
                freqs[col] = float(tag_freqs.get(tag, 0))
        order = torch.argsort(freqs, descending=True)
        decile_ids = torch.empty(c_metric, dtype=torch.long)
        # Equal-count buckets; remainder spreads over the leading buckets.
        base, rem = divmod(c_metric, self.num_deciles)
        start = 0
        for d in range(self.num_deciles):
            size = base + (1 if d < rem else 0)
            decile_ids[order[start:start + size]] = d
            start += size
        self.decile_ids_cpu = decile_ids
        self.decile_ids = decile_ids.to(self.device)

        # --- Rating-tag exclusion mask (True = counted in dp metrics) ---
        content = torch.ones(c_metric, dtype=torch.bool)
        if getattr(t, "exclude_rating_tags", True):
            n_rating = 0
            for col in range(c_metric):
                tag = vocab.index_to_tag.get(col + SKIP_COLS)
                if isinstance(tag, str) and tag.startswith(RATING_PREFIX):
                    content[col] = False
                    n_rating += 1
            if n_rating:
                logger.info(
                    "ASL telemetry: excluding %d rating tags from dp/top-K metrics "
                    "(SS5 hygiene: they inflate mean(p_pos)).", n_rating,
                )
        self.content_mask_cpu = content
        self.content_mask = content.to(self.device)
        # Float copies for the per-decile EPR column sums. EPR must honour the
        # same exclusion as dp_mean/dp_hard: every training row carries exactly
        # one rating positive, so with the rating tags left in they dominate the
        # denominator of whichever decile they land in (in the shipped vocab they
        # have no entry in tag_frequencies, so they sort to the RAREST decile and
        # make up ~87% of its expected-positive mass). That desensitises the
        # per-decile EPR trend -- the primary always-on over-suppression gate --
        # in exactly the decile a high gamma_neg is most likely to damage.
        self.content_mask_f = self.content_mask.float()
        self.content_mask_f_cpu = content.float()

        # --- Confusable sibling groups (val-side sibling-gap metric) ---
        self.sibling_groups: List[Tuple[str, torch.Tensor]] = []
        path = getattr(t, "sibling_groups_path", None)
        if path:
            p = Path(path)
            if p.exists():
                try:
                    raw = json.loads(p.read_text(encoding="utf-8"))
                    for name, tags in raw.items():
                        if str(name).startswith("_") or not isinstance(tags, list):
                            continue  # metadata keys like "_comment"
                        idx = [
                            vocab.tag_to_index[tag] - SKIP_COLS
                            for tag in tags
                            if tag in vocab.tag_to_index
                            and vocab.tag_to_index[tag] >= SKIP_COLS
                        ]
                        if len(idx) >= 2:
                            self.sibling_groups.append(
                                (str(name), torch.tensor(idx, dtype=torch.long))
                            )
                        else:
                            logger.debug(
                                "ASL telemetry: sibling group '%s' has <2 tags in "
                                "vocab; skipped.", name,
                            )
                    logger.info(
                        "ASL telemetry: %d confusable sibling groups loaded from %s.",
                        len(self.sibling_groups), p,
                    )
                except Exception as e:
                    logger.warning("ASL telemetry: failed to load sibling groups from %s: %s", p, e)
            else:
                logger.warning(
                    "ASL telemetry: sibling_groups_path %s not found -- sibling-gap "
                    "metric disabled.", p,
                )

        # --- EMA state (GPU tensors; serialized to floats at logging cadence) ---
        self._dp_mean_ema = torch.zeros((), device=self.device)
        self._dp_hard_ema = torch.zeros((), device=self.device)
        self._epr_num_ema = torch.zeros(self.num_deciles, device=self.device)
        self._epr_den_ema = torch.zeros(self.num_deciles, device=self.device)
        self._ema_ready = False
        self._last_alarm_epoch = None

        tele_state = self.state.get("telemetry")
        if isinstance(tele_state, dict) and tele_state.get("epr_num"):
            try:
                self._dp_mean_ema.fill_(float(tele_state.get("dp_mean", 0.0)))
                self._dp_hard_ema.fill_(float(tele_state.get("dp_hard", 0.0)))
                num = tele_state.get("epr_num") or []
                den = tele_state.get("epr_den") or []
                if len(num) == self.num_deciles and len(den) == self.num_deciles:
                    self._epr_num_ema.copy_(torch.tensor(num, device=self.device))
                    self._epr_den_ema.copy_(torch.tensor(den, device=self.device))
                    self._ema_ready = True
                    logger.info("ASL telemetry: EMA state restored from checkpoint.")
            except Exception as e:
                logger.warning("ASL telemetry: could not restore EMA state: %s", e)
        self.state.setdefault("telemetry", {})

    @torch.no_grad()
    def on_update(self, tag_logits: torch.Tensor, tag_labels: torch.Tensor,
                  global_step: int, epoch0: int) -> None:
        """Train-side sample. Call ONLY on optimizer-update boundaries (SS5
        hygiene: sampling inside an accumulation window aliases the EMA).
        ``tag_logits`` must be detached, full-width (PAD/UNK included)."""
        if not self.enabled or (global_step % self.interval) != 0:
            return

        probs = torch.sigmoid(tag_logits[:, SKIP_COLS:].float())  # fp32 upcast (SS5)
        targs = tag_labels[:, SKIP_COLS:] > 0.5

        m = self.content_mask
        pos = targs & m
        neg = (~targs) & m
        pos_cnt = pos.sum()
        neg_cnt = neg.sum()
        if pos_cnt == 0 or neg_cnt == 0:
            return

        pos_mean = (probs * pos).sum() / pos_cnt
        neg_mean = (probs * neg).sum() / neg_cnt
        dp_mean = pos_mean - neg_mean

        # top-K non-GT capture excludes PAD/UNK (already sliced) and rating cols
        k = min(self.topk, probs.size(1))
        top = probs.masked_fill(targs | ~m, -1.0).topk(k, dim=1).values
        dp_hard = pos_mean - top.clamp(min=0.0).mean()

        # Threshold-free EPR components per tag-frequency decile (Cole 2021):
        # EMA numerator (sum p) and denominator (expected positives) separately
        # so rare deciles with zero positives in a single batch stay stable.
        # Rating columns are zeroed out of both sums (see content_mask_f).
        mf = self.content_mask_f
        epr_num = torch.zeros(self.num_deciles, device=probs.device)
        epr_num.scatter_add_(0, self.decile_ids, probs.sum(dim=0) * mf)
        epr_den = torch.zeros(self.num_deciles, device=probs.device)
        epr_den.scatter_add_(0, self.decile_ids, targs.float().sum(dim=0) * mf)

        if not self._ema_ready:
            self._dp_mean_ema.copy_(dp_mean)
            self._dp_hard_ema.copy_(dp_hard)
            self._epr_num_ema.copy_(epr_num)
            self._epr_den_ema.copy_(epr_den)
            self._ema_ready = True
        else:
            b = self.beta
            self._dp_mean_ema.mul_(b).add_(dp_mean, alpha=1 - b)
            self._dp_hard_ema.mul_(b).add_(dp_hard, alpha=1 - b)
            self._epr_num_ema.mul_(b).add_(epr_num, alpha=1 - b)
            self._epr_den_ema.mul_(b).add_(epr_den, alpha=1 - b)

        if (global_step % self.log_interval) == 0:
            self._log_train(global_step, epoch0)

    def _log_train(self, global_step: int, epoch0: int) -> None:
        # One packed D2H transfer instead of many .item() syncs
        packed = torch.cat([
            self._dp_mean_ema.reshape(1),
            self._dp_hard_ema.reshape(1),
            self._epr_num_ema,
            self._epr_den_ema,
        ]).cpu().tolist()
        dp_mean, dp_hard = packed[0], packed[1]
        num = packed[2:2 + self.num_deciles]
        den = packed[2 + self.num_deciles:]
        epr = [n / d if d > 1e-9 else float("nan") for n, d in zip(num, den)]

        scalars = {
            "asl/gamma_neg": float(self.gamma),
            "asl/dp_mean": dp_mean,
            "asl/dp_hard": dp_hard,
        }
        for d, v in enumerate(epr):
            if math.isfinite(v):
                scalars[f"asl/epr_decile_{d:02d}"] = v

        # Shadow controller (SS4: logging only, zero authority)
        if self.shadow_enabled:
            shadow = self.gamma + self.shadow_lambda * (self.shadow_target - dp_mean)
            if self.window is not None:
                shadow = min(max(shadow, float(self.window.gamma_neg_min)),
                             float(self.window.gamma_neg_max))
            scalars["asl/gamma_neg_shadow"] = shadow

        # SS5 EPR trend alarm: sustained relative drop in any decile within
        # epr_alarm_window epochs of a gamma step -> step back up.
        alarm = 0.0
        # Deferred baseline capture: a gamma step taken before any EPR sample
        # existed left a marker instead of a snapshot. Fill it from the first
        # real EPR we compute, anchored to the epoch of the step.
        pending = self.state.pop("epr_baseline_pending_epoch", None)
        if pending is not None and any(math.isfinite(v) for v in epr):
            self.state["epr_baseline"] = list(epr)
            self.state["epr_baseline_epoch"] = int(pending)
            logger.info(
                "ASL telemetry: captured deferred EPR trend-alarm baseline for the "
                "epoch-%d gamma step.", int(pending),
            )
        elif pending is not None:
            self.state["epr_baseline_pending_epoch"] = pending  # still nothing usable

        baseline = self.state.get("epr_baseline")
        base_epoch = self.state.get("epr_baseline_epoch")
        # `0 <=` matters: without a lower bound a baseline whose epoch is AHEAD of
        # the current one (only reachable if a phase-local baseline survived a
        # phase transition) satisfies the window with a negative elapsed and
        # alarms indefinitely. _reconcile_gamma now drops those, this is belt-and-braces.
        elapsed_epochs = (epoch0 + 1) - int(base_epoch) if base_epoch is not None else None
        if baseline and elapsed_epochs is not None and 0 <= elapsed_epochs <= self.epr_alarm_window:
            for d, (cur, ref) in enumerate(zip(epr, baseline)):
                if (ref is not None and math.isfinite(cur) and math.isfinite(ref)
                        and ref > 1e-9 and (ref - cur) / ref > self.epr_alarm_rel_drop):
                    alarm = 1.0
                    if self._last_alarm_epoch != epoch0:
                        self._last_alarm_epoch = epoch0
                        logger.warning(
                            "ASL EPR ALARM: decile %d EPR dropped %.1f%% vs the "
                            "baseline captured at the epoch-%s gamma step (%.4f -> "
                            "%.4f). ASL_plan SS5: step gamma_neg back up.",
                            d, 100 * (ref - cur) / ref, base_epoch, ref, cur,
                        )
        scalars["asl/epr_alarm"] = alarm

        if self.monitor is not None:
            for tag, value in scalars.items():
                self.monitor.log_scalar(tag, value, global_step)

        # Persist floats into the checkpoint-bound state dict
        self.state["telemetry"] = {
            **(self.state.get("telemetry") or {}),
            "dp_mean": dp_mean,
            "dp_hard": dp_hard,
            "epr_deciles": epr,
            "epr_num": num,
            "epr_den": den,
            "step": int(global_step),
        }
        logger.debug(
            "ASL telemetry @%d: gamma=%.3f dp_mean=%.4f dp_hard=%.4f", global_step,
            self.gamma, dp_mean, dp_hard,
        )

    @torch.no_grad()
    def compute_val(self, cat_probs: torch.Tensor, cat_targs: torch.Tensor,
                    global_step: int, epoch0: int, chunk_rows: int = 2048) -> None:
        """Val-side SS5 set: consumes the accumulated CPU prob/target matrices
        (full-width, PAD/UNK included). Pure consumer -- no extra GPU work."""
        if not self.enabled or cat_probs is None or cat_targs is None:
            return

        n = cat_probs.size(0)
        m = self.content_mask_cpu
        mf = self.content_mask_f_cpu
        ids = self.decile_ids_cpu
        k = self.topk

        pos_sum = 0.0
        pos_cnt = 0
        neg_sum = 0.0
        neg_cnt = 0
        top_sum = 0.0
        top_cnt = 0
        hist = torch.zeros(self.hist_bins)
        epr_num = torch.zeros(self.num_deciles)
        epr_den = torch.zeros(self.num_deciles)
        gap_sums = {name: 0.0 for name, _ in self.sibling_groups}
        gap_cnts = {name: 0 for name, _ in self.sibling_groups}

        for i in range(0, n, chunk_rows):
            p = cat_probs[i:i + chunk_rows, SKIP_COLS:].float()
            t = cat_targs[i:i + chunk_rows, SKIP_COLS:].bool()

            pos = t & m
            neg = (~t) & m
            pos_sum += float((p * pos).sum())
            pos_cnt += int(pos.sum())
            neg_sum += float((p * neg).sum())
            neg_cnt += int(neg.sum())

            # Rating columns are pushed to -1, which falls outside [hist_min,
            # hist_max] so torch.histc drops them, and below the topk floor so
            # they can never be selected as a hard non-GT capture.
            p_c = p.masked_fill(~m, -1.0)

            kk = min(k, p.size(1))
            top = p_c.masked_fill(t, -1.0).topk(kk, dim=1).values
            top_sum += float(top.clamp(min=0.0).sum())
            top_cnt += top.numel()

            # Non-GT histogram = hist(all) - hist(GT); GT entries are sparse.
            hist += torch.histc(p_c, bins=self.hist_bins, min=self.hist_min, max=self.hist_max)
            hist -= torch.histc(p_c[t], bins=self.hist_bins, min=self.hist_min, max=self.hist_max)

            # Same rating exclusion as the train side (see content_mask_f).
            epr_num.scatter_add_(0, ids, p.sum(dim=0) * mf)
            epr_den.scatter_add_(0, ids, t.float().sum(dim=0) * mf)

            for name, gidx in self.sibling_groups:
                sub_p = p[:, gidx]
                sub_t = t[:, gidx]
                one = sub_t.sum(dim=1) == 1
                cnt = int(one.sum())
                if cnt == 0:
                    continue
                labeled = (sub_p * sub_t).sum(dim=1)[one]
                unlabeled_max = sub_p.masked_fill(sub_t, -1.0).max(dim=1).values[one]
                gap_sums[name] += float((labeled - unlabeled_max).sum())
                gap_cnts[name] += cnt

        hist = hist.clamp(min=0)
        hist_total = float(hist.sum())
        edges = torch.linspace(self.hist_min, self.hist_max, self.hist_bins + 1)
        band = torch.zeros(self.hist_bins, dtype=torch.bool)
        for b in range(self.hist_bins):
            if edges[b] >= self.band_low - 1e-9 and edges[b + 1] <= self.band_high + 1e-9:
                band[b] = True
        band_count = float(hist[band].sum())

        scalars: Dict[str, float] = {"asl_val/gamma_neg": float(self.gamma)}
        if pos_cnt and neg_cnt:
            dp_mean = pos_sum / pos_cnt - neg_sum / neg_cnt
            scalars["asl_val/dp_mean"] = dp_mean
        if pos_cnt and top_cnt:
            scalars["asl_val/dp_hard"] = pos_sum / pos_cnt - top_sum / top_cnt
        if hist_total > 0:
            # band_frac: share of non-GT scores in [hist_min, hist_max] that sit
            # in the clip watch band -- the SS2 clip-cost observable.
            scalars["asl_val/nongt_band_count_per_img"] = band_count / max(1, n)
            scalars["asl_val/nongt_band_frac"] = band_count / hist_total
            for b in range(self.hist_bins):
                scalars[f"asl_val/nongt_hist/{edges[b]:.2f}"] = float(hist[b]) / max(1, n)
        epr_list = []
        for d in range(self.num_deciles):
            v = float(epr_num[d] / epr_den[d]) if float(epr_den[d]) > 1e-9 else float("nan")
            epr_list.append(v)
            if math.isfinite(v):
                scalars[f"asl_val/epr_decile_{d:02d}"] = v
        gaps = {}
        for name, _ in self.sibling_groups:
            if gap_cnts[name] > 0:
                g = gap_sums[name] / gap_cnts[name]
                gaps[name] = g
                scalars[f"asl_val/sibling_gap/{name}"] = g
        if gaps:
            scalars["asl_val/sibling_gap_macro"] = sum(gaps.values()) / len(gaps)

        if self.monitor is not None:
            for tag, value in scalars.items():
                self.monitor.log_scalar(tag, value, global_step)

        tele = dict(self.state.get("telemetry") or {})
        tele["val"] = {
            "step": int(global_step),
            "epoch": int(epoch0) + 1,
            "dp_mean": scalars.get("asl_val/dp_mean"),
            "dp_hard": scalars.get("asl_val/dp_hard"),
            "band_frac": scalars.get("asl_val/nongt_band_frac"),
            "epr_deciles": epr_list,
            "sibling_gaps": gaps,
        }
        self.state["telemetry"] = tele

        summary = ", ".join(
            f"{k.split('/', 1)[1]}={v:.4f}" for k, v in scalars.items()
            if k in ("asl_val/dp_mean", "asl_val/dp_hard", "asl_val/nongt_band_frac",
                     "asl_val/sibling_gap_macro")
        )
        logger.info("ASL val telemetry (gamma_neg=%.3f): %s", self.gamma, summary or "n/a")
