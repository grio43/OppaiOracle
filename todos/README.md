# OppaiOracle V2 roadmap

Start with this map, then use the owning plan for implementation details. The production
Phase 1 recipe is implemented; the second dataset subset, preparation, and full-scale launch checks remain.
Research recommendations remain proposals until recorded as decisions in the owning plan.

## Current plans and ownership

| Document | Owns | Work queue / status |
|---|---|---|
| [V2 production plan](v2-plan.md) | Architecture, fixed ASL loss, three-phase WSD recipe, production splits, selection and release gates | **§12** is the production implementation and launch checklist; recipe decisions are settled |
| [Phase 1 setup](v2-phase1-setup.md) | Preparation/launch commands, current dataset layout, rating migration, implemented P1 scope | Wait for the additional subset before building vocabulary; 60-epoch ceiling includes cooldown |
| [Training monitoring](v2-monitoring.md) | Per-validation procedure, canaries, operational diagnostics | Re-derive V1 numeric bands on V2; implement the adopted augmentation AP slices |
| [Augmentation decisions](v2-augmentation.md) | Consolidated status of both reviews, pending D1–D4 decisions, downstream routing | Production settings remain in **v2-plan §7**; proposals are not defaults |
| [Janitor cleaning campaign](janitor-cleaning-model-plan.md) | Fresh Stage A, frozen-backbone refits, calibration, corpus scoring, certification, reversible writeback | **§8** is the campaign queue; **§6** owns writeback and discharge gates |
| [V2.1.1 gold-training pipeline](v2.1.1-gold-training-pipeline.md) | Full training on the gold-augmented corpus, off-diagonal metric, prior correction, acceptance criteria | **§3–7** hold prerequisites and open decisions; **§9** defines acceptance |

The production plan governs shared recipe decisions. Janitor §2 explicitly lists its
campaign-specific deltas, including its EMA copy; these do not reintroduce EMA into production
V2. V2.1.1 owns the off-diagonal metric specification (§4), even though the janitor needs that
implementation first. Detailed tasks stay with their owner; this roadmap supplies the order.

## Order of work

1. **Make the shared training path ready.** Work through [V2 §12](v2-plan.md#12-order-of-work),
   including the precision, holdout, resume, checkpoint, loss, and stop-policy issues linked
   to the September audit. Verify implementation against the decided recipe before either
   production P1 or janitor Stage A.
2. **Prepare real review data alongside that work.** V2 §9.4 owns the production gold sampling protocol; freeze it before P1 and
   assemble the V2-dependent judgment pool after fixed-model inference. Janitor §3 owns its real slice and TUNE/CAL/SEALED manifests, frozen before Stage A.
   Share review infrastructure; preserve each plan's sampling and split rules.
3. **Freeze the production experiment.** Freeze the actual TRAIN/30K EVAL labels and baseline
   under V2 §9.3's current-launch procedure; complete §12 budgets,
   and monitoring preparation. Any proposed augmentation adoption must first amend §7;
   pending proposals do not silently change the existing recipe.
4. **Run each campaign against its own gates.** Production follows P1 → P2 → P3 using the
   same 30K validation holdout throughout. The larger CALIB/TEST allocation and full three-gate
   release comparison remain deferred. Janitor follows Stage A → B′ → C → D → E → F, with the early pinned
   diagnostic checkpoint and the cooled branch checkpoint specified in its §2.
5. **Review cleaning results, then train downstream.** Janitor certification precedes
   automatic writeback; discharge review precedes the next production launch. V2.1.1 consumes
   corrected data, with its regime, synthetic fraction, and ordinal scope decided in §7.
   If V2.1.1 runs before cleaning, follow its §6 sequencing rule for prior correction.

This is a dependency order, not a commitment to run the two expensive P1 campaigns in
parallel or to merge their checkpoints.

## Data boundaries to preserve

| Artifact | Owner and purpose |
|---|---|
| EVAL / CALIB / TEST / TAGSET | V2 §8.1: current holdout is 30K EVAL only; excess candidates return to training. Additional CALIB/TEST reservations are deferred by user decision |
| GOLD-SLICE | V2 §9.4: protocol/candidate universe frozen before P1; realized two-model pool hashed after fixed V2 inference, then independently adjudicated for Gate 3 |
| TUNE / CAL / SEALED | Janitor §3.2: refitting, calibration, and qualifying gates; CAL and SEALED are real-only |
| Shared real evaluation slice | V2.1.1 §3 consumes the janitor SEALED split; keep it disjoint from downstream training |
| Anima gold | Tuning/training input only, never calibration, evaluation, certification, or a release arbiter |

Shared review infrastructure does not make GOLD-SLICE and SEALED interchangeable. Follow the
owning sampling protocol. Gold taxonomy/rubrics remain in
[golden_set_plan](../.research/golden_set_plan.md); collection numbers and provenance remain in
[golden_set_collection_targets](../.research/golden_set_collection_targets.md). These research
files are local, git-ignored references and may be absent in a fresh checkout.

## Reviews and history

| Evidence | How to use it |
|---|---|
| [V2 evidence appendix](reviews/v2-plan-review-2026-07-28.md) | Frozen measurements and citations; the production plan wins on disagreements |
| [Augmentation round 1](reviews/v2-augmentation-review.md) and [round 2](reviews/v2-augmentation-review-round2.md) | Detailed research and provenance; read the [consolidated decision record](v2-augmentation.md) first |
| [ViT implementation audit](reviews/vit-major-issues-audit-2026-09-05.md) | Working-tree findings, reproductions, closure criteria, and verification limits; issue IDs are mapped into V2 §12 |
| [V2 training readiness review](reviews/v2-training-readiness-review-2026-09-06.md) | Normal-startup review after the fixes: cache-order resume, bad-image exclusions, and RTX 5090 resource measurements |
| [V2 scholarly plan review](reviews/v2-plan-scholarly-review-2026-09-06.md) | Remaining baseline, gold-label, boundary-metric, calibration, and cleaning-certification gaps; primary/expert sources and CPU mathematical checks; recommendations are not adopted decisions |
| [Pre-rewrite V2 snapshot](archive/v2-plan-pre-rewrite-2026-07-30.md) | Historical plan, never a launch specification |
| [V1 training health archive](../TRAINING_HEALTH_TRACKER.md) | Historical epoch measurements; monitoring procedure lives in v2-monitoring.md |

Retired plans remain in git history: `progressive-training-plan.md`, `ASL_plan.md`, and
`v2-plan-correction-2026-07-28.md`. The deferred ordinal-fusion specification is recoverable
with `git show 2c1cfa1:todos/ordinal-fusion-track.md`; its living summary is janitor §7.1.

## Editing rules

- Amend the owning plan in place; avoid new dated correction plans or duplicate task lists.
- Keep adopted decisions, pending proposals, and historical findings explicitly separate.
- Propagate production P1 recipe amendments into janitor §2; preserve its explicit deltas.
- Put supporting reviews in `reviews/` and superseded snapshots in `archive/`; preserve their
  evidence and dates. Update links when moving files.
- A checked-off task needs implementation evidence. This document reorganization does not
  mark code changes, training runs, or research proposals complete.
