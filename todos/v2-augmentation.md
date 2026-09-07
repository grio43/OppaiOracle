# V2 augmentation — consolidated decision record

[Roadmap](README.md) · [Production recipe §7](v2-plan.md#7-augmentation--unchanged-and-defensible)
· [Monitoring](v2-monitoring.md)

This file consolidates the decision status and follow-ups from the two augmentation reviews.
**The adopted per-phase settings live only in v2-plan §7.** A later research recommendation
does not override an adopted decision. Round 2 refines the research assessment of round 1;
it does not approve a new production configuration.

Evidence: [round 1](reviews/v2-augmentation-review.md) (2026-07-31) and
[round 2](reviews/v2-augmentation-review-round2.md) (2026-08-07). Their detailed mechanisms,
candidate tables, citations, caveats, and original recommendations remain available there.

## Adopted policy

- Keep the existing flip, color-jitter, rotation, and blur recipe in V2 §7, including its
  phase-specific strengths. No additional augmentation has been adopted through this consolidation.
- Translation and downscale-only letterbox jitter are excluded from production V2 and routed
  by name to V2.1.1 for consideration; routing is not adoption into that training run.
- Consistency regularization / teacher-model proposals were declined. Production weight EMA
  is also cut (V2 §6); janitor's explicit EMA delta is a separate campaign requirement.
- No mid-run augmentation additions or experiment arms. Same-phase augmentation compatibility
  checks are warning-tier; incompatible critical resume settings must fail loudly (V2 §12).
- Luminance-pair and letterbox/border AP monitoring was adopted in V2 §7. Its operational
  specification is now in [the monitoring plan](v2-monitoring.md#per-tag-group-ap-slices).

**Flip correction:** the pipeline has no directional-tag swap. Round 2 §5.1 identifies
`left-handed`, `left-to-right_manga`, `right-over-left_kimono`, and `right-to-left_comic` as
chirality exceptions. Flip remains at its adopted probability; ignoring these tags is **D1,
still pending**, and must not be implied by calling the current flip lossless.

## Pending decisions — not approved

| ID | Question | Latest research recommendation | Source |
|---|---|---|---|
| D1 | Ignore the four chirality tags? | Adopt name-based resolution into `ignore_indices`, never frozen vocabulary indices; align metric exclusions | Round 2 §5.1 |
| D2 | Add sharpness / unsharp-mask jitter? | Conditional on all G1–G5 gates: actual consumer, measured degraded-input penalty, step-0 amendment, shared detail budget, per-tag checks, and accurate config documentation | Round 2 §2.1 |
| D3 | Add true random token removal? | Decline for V2; retain as a possible later occlusion route. The mechanism remains conditional, not approved | Round 2 §2.2 |
| D4 | Reopen Phase-1 intensity direction? | Only by explicitly reopening V2 §7 against the actual P1 baseline; no automatic strength reduction | Round 2 §3.2 |

Record a decision in the owning plan before implementing any of these. The existing recipe
remains the baseline if none is adopted; this list is not an instruction to reopen settled
architecture, loss, or scheduling decisions.

## Candidate disposition after both reviews

The statuses below describe the latest **research assessment**. Conditional and parked items
remain outside the production recipe, and review rejections are not new user decisions.

| Candidate | Consolidated disposition | What would be needed to revisit it |
|---|---|---|
| Letterbox translation; downscale-only scale jitter | Routed to V2.1.1, not adopted; round 2 recommends parking and not bundling | An explicitly approved, separately attributable augmentation study; translation first if reopened |
| P2/P3 rotation widening to ±7–8° | Round-1 recommendation overturned by round 2: do not add | Reopen §7 explicitly; current P1 and P2/P3 ranges retain their different purposes |
| JPEG recompression | Conditional, tightened in round 2 | Name a consumer presenting recompressed inputs, then measure degradation at that consumer's quality distribution |
| Downscale-upscale degradation | Conditional, unchanged | Round-1 §3.3 gates and the degraded-EVAL harness |
| Sharpness / unsharp mask | Conditional, D2 | All round-2 §2.1 G1–G5 gates; mutually exclusive with blur per sample |
| Random token removal | Conditional mechanism; recommend decline for V2, D3 | Explicit adoption and round-2 §2.2 safety, phase, mask, and implementation prerequisites |
| Random Erasing | Round 2 rejects for V2 | Current single-config discipline cannot host its monitor-and-abort conditions |
| Gamma / tone-curve jitter | Round 2 rejects for V2 | Audit the existing luminance exposure and demonstrate a tone deficit first |
| Restricted TrivialAugment | Round 2 rejects for V2 | Resolve composed-stack safety and attribution requirements |
| WSD cooldown augmentation annealing | Round 2 rejects for V2 | Requires an ablation arm that the production plan excludes |
| Deterministic pad-token removal | Parked throughput work, not augmentation | A separate token-removal or aspect-bucketing implementation makes it relevant |
| P1 intensity schedule | Parked, D4 | Re-derive against the actual stronger P1 column, with total intensity controlled |

Other rejected families and their mechanisms remain in round-1 §6 and round-2 §4. Consult
those tables before proposing another sweep; preserving the evidence avoids repeating the
same candidate arguments.

## Implementation and documentation follow-ups

| Work | Owner | Status |
|---|---|---|
| Rotation-mask regression coverage, fixed detached-ASL conformance, augmentation resume warnings | V2 §12 | Existing implementation checklist; September audit records passing working-tree rotation tests, not a committed release |
| Adopted luminance-pair and letterbox/border AP slices | v2-monitoring | Specification consolidated; logging still needs implementation |
| Chirality handling | D1 → V2 §7 and metric/loss configuration | Pending decision; false lossless/tag-swap wording corrected in the plan |
| Degraded-EVAL harness for JPEG, sharpness, down-up copies | Round-1 §4.3; round-2 §8 | Proposed enabling tool; no training augmentation is authorized by building it |
| Default-value alignment and explicit validation blur disable | Round-2 §5.6 | Proposed code hardening; retain phase-specific recipe values |
| Stale BYOL/saturation and blur comments; false orientation-agnostic code comment | Round-2 §5.1–5.5 | Proposed source/config documentation cleanup |
| Extra AP groups: hue controls, script text, `dutch_angle`, blur family | Round-2 §5.7 → monitoring plan | Pending expansion; distinct from the already adopted AP riders |
| Brightness/contrast fallback 0.22→0.15 / 0.15→0.10 at a phase boundary | Round-2 §5.2 → V2 §7 | Proposed fallback, not active policy |
| Never photometrically perturb the pad; specify a shared detail-operation budget | Round-2 §7.1–7.2 → V2 §7 on adoption | Review requirements for future changes; no new implementation claimed |

For any future adoption, carry both ASL failure channels: removing evidence under a retained
positive label still trains that positive; rendering an unlabeled in-vocabulary artifact
punishes its correct detection on the negative branch. Audit vocabulary mimicry and composed
severity together. The reviews' analytic color-exposure estimates are not corpus measurements,
and V1's lack of a recorded pair collapse is not proof that those pairs were measured.
