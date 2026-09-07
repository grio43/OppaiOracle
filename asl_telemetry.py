"""Fixed ASL contract and training/validation telemetry.

The V2 objective is checked after checkpoint loading. Gamma is fixed at 7,
focal weights are detached, clip is .05, and checkpoint state cannot override
the configured objective. The old manual gamma driver has no authority.
Telemetry state is checkpointed; the shadow controller is diagnostic only.

Measurement hygiene (SS5): all metrics are computed on columns >= 2 (PAD=0
and UNK=1 are live, loss-free, drifting outputs), probabilities are fp32
upcast before sigmoid, and rating tags are excluded from EVERY column-space
metric -- dp_mean, dp_hard, the non-GT histogram AND the per-decile EPR.
Every rated row carries one rating positive, so leaving them in inflates
mean(p_pos) and swamps the EPR denominator of whichever decile they sort into.
Unknown targets are always excluded, even when rating diagnostics are enabled.

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
    """Verifies fixed ASL and owns the checkpointed telemetry state.

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

    # Telemetry counters/baselines are phase-local and reset at transitions.
    _PHASE_LOCAL_STATE_KEYS = (
        "gamma_last_change_epoch",
        "epr_baseline",
        "epr_baseline_epoch",
        "epr_baseline_pending_epoch",
        "telemetry",
    )

    def _reconcile_gamma(self, tag_loss_cfg, start_epoch: int) -> None:
        """Enforce the fixed V2 objective after checkpoint loading."""
        expected = dict(gamma_neg=7.0, gamma_pos=0.0, clip=0.05, alpha=1.0,
                        label_smoothing=0.0, detach_focal_weight=True)
        mismatches = []
        loss_fn = self.criterion.tag_loss_fn
        if loss_fn.reduction != 'mean' or sorted(loss_fn.ignore_indices) != [0, 1]:
            mismatches.append("loss must use mean reduction and ignore exactly PAD/UNK [0, 1]")
        for name, value in expected.items():
            for origin, obj in (("config", tag_loss_cfg), ("criterion", loss_fn)):
                if getattr(obj, name, None) != value:
                    mismatches.append(f"{origin}.{name}={getattr(obj, name, None)!r}, expected {value!r}")
        if getattr(tag_loss_cfg, "gamma_neg_override", None) is not None:
            mismatches.append("gamma_neg_override must be null (fixed loss)")
        if self.sched is not None and self.sched.enabled:
            mismatches.append("asl_schedule.enabled must be false (fixed loss)")
        if loss_fn.class_weights is not None or tag_loss_cfg.class_weight_strategy is not None:
            mismatches.append("class weighting must be disabled (plain ASL)")
        persisted = self.state.get("gamma_neg")
        if persisted is not None and float(persisted) != 7.0:
            mismatches.append(f"checkpoint gamma_neg={persisted}, expected 7.0")
        if mismatches:
            raise ValueError("Fixed ASL contract violated: " + "; ".join(mismatches))
        if self.state.get("phase") != self.phase:
            for key in self._PHASE_LOCAL_STATE_KEYS:
                self.state.pop(key, None)
        self.gamma = 7.0
        self.state.update(gamma_neg=self.gamma, phase=self.phase)
        self.state.setdefault("gamma_history", [])
        self.criterion.set_gamma_neg(self.gamma)
        logger.info("Fixed ASL verified: gamma_neg=7, gamma_pos=0, clip=.05, detached focal weights")

    def request_gamma_step(self, target: float, epoch0: int, source: str = "manual") -> bool:
        """The retired gamma controller has no mutation authority."""
        logger.error("Refusing gamma step: the training loss is fixed at gamma_neg=7")
        return False

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
        # EPR uses the same configured rating exclusion as dp_mean/dp_hard.
        # Rated images have one rating positive; unknown labels are masked per
        # image below even when rating diagnostics are explicitly enabled.
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
                        tags = [f"gen:{tag}" if f"gen:{tag}" in vocab.tag_to_index else tag for tag in tags]
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
        observed = tag_labels[:, SKIP_COLS:] >= 0

        m = self.content_mask & observed
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
        epr_num.scatter_add_(0, self.decile_ids, probs.masked_fill(~observed, 0).sum(dim=0) * mf)
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
                    global_step: int, epoch0: int,
                    chunk_rows: int = 2048) -> Dict[str, float]:
        """Val-side SS5 set: consumes the accumulated CPU prob/target matrices
        (full-width, PAD/UNK included). Pure consumer -- no extra GPU work.

        Returns the scalar dict it logged (empty when disabled), so callers can
        read a value directly; see the note at the return statement.
        """
        if not self.enabled or cat_probs is None or cat_targs is None:
            return {}

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
            t = cat_targs[i:i + chunk_rows, SKIP_COLS:] > 0.5
            observed = cat_targs[i:i + chunk_rows, SKIP_COLS:] >= 0

            pos = t & m
            neg = (~t) & m & observed
            pos_sum += float((p * pos).sum())
            pos_cnt += int(pos.sum())
            neg_sum += float((p * neg).sum())
            neg_cnt += int(neg.sum())

            # Rating columns are pushed to -1, which falls outside [hist_min,
            # hist_max] so torch.histc drops them, and below the topk floor so
            # they can never be selected as a hard non-GT capture.
            p_c = p.masked_fill(~m | ~observed, -1.0)

            kk = min(k, p.size(1))
            top = p_c.masked_fill(t, -1.0).topk(kk, dim=1).values
            top_sum += float(top.clamp(min=0.0).sum())
            top_cnt += top.numel()

            # Non-GT histogram = hist(all) - hist(GT); GT entries are sparse.
            hist += torch.histc(p_c, bins=self.hist_bins, min=self.hist_min, max=self.hist_max)
            hist -= torch.histc(p_c[t], bins=self.hist_bins, min=self.hist_min, max=self.hist_max)

            # Same rating exclusion as the train side (see content_mask_f).
            epr_num.scatter_add_(0, ids, p.masked_fill(~observed, 0).sum(dim=0) * mf)
            epr_den.scatter_add_(0, ids, t.float().sum(dim=0) * mf)

            for name, gidx in self.sibling_groups:
                sub_p = p[:, gidx]
                sub_t = t[:, gidx]
                one = (sub_t.sum(dim=1) == 1) & observed[:, gidx].all(dim=1)
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
        # Returned so the caller can consume a scalar without re-reading
        # TensorBoard. The stop system takes `asl_val/sibling_gap_macro` from here
        # as its clean-label arbiter: a sibling-positive label is reliable
        # evidence of negativity for the rest of the group, so the gap is a
        # low-noise ranking margin, immune to the missing-positive bias that makes
        # val_mAP fall as the model outgrows the annotation (v2-plan.md SS9.1/SS9.2).
        # Callers that ignore the return value are unaffected.
        return scalars
