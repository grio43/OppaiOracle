# V2 plan review — scholarly evidence, 2026-09-06

The most consequential remaining gaps are in the evaluation definitions and cleaning
certification. Several could make a worse boundary model appear better, or let a cleaning
rule pass without the claimed precision guarantee. The production recipe has already
corrected many of the earlier architecture/loss overclaims; the downstream plans have
not consistently inherited those corrections.

Scope: the working tree at `ad17ee9`, including existing uncommitted changes; V2 production,
Phase 1 setup, monitoring, augmentation decisions, janitor, V2.1.1, and local gold-design
references. Online sources checked on 2026-09-06 are papers, author/institutional copies,
conference proceedings, and statistical authorities. **No external anime-tagger
documentation or anime-tagger papers were used as evidence.** Historical measurements
quoted by the plans were not remeasured on the unfinished production corpus.

This is a supporting review, not an adopted recipe amendment. P1 below means resolve
before relying on the affected stage or claim; it does not mean every downstream issue
blocks production Phase 1. No production code/configuration, dataset, vocabulary, or
checkpoint was changed. The 30K-only holdout, single training configuration, from-scratch
decision, and upstream deduplication contract remain the review constraints.

| ID | Priority | Affected decision | Finding |
|---|---|---|---|
| R1 | P1 | Production baseline comparison | The new 30K draw does not preserve the old V1-clean baseline; artifact and vocabulary alignment are unspecified |
| R2 | P1 | Shared boundary metric | Conditional cross-fire counts legitimate co-occurrence and implication as error |
| R3 | P1 | Downstream evaluation/certification | SEALED and audit labels are also assigned to fitting and selection |
| R4 | P1 | Janitor CAL/SEALED construction | Routing all disagreements into TUNE removes a critical population from evaluation |
| R5 | P1 | Automatic cleaning | Fixed stratum quotas cannot be pooled into the stated population binomial bound |
| R6 | P1 | Automatic cleaning | Reviewer-error correction and threshold selection lack the claimed coverage guarantee |
| R7 | P1 | Prior correction | The sigmoid offset is mathematically incomplete and the deployment prior is not known |
| R8 | P1 | Gold training and broad regression checks | Axis-only labels lack an end-to-end observation-mask contract |
| R9 | P2 | Janitor missing-positive recall | The early scorer cannot recover candidates excluded by the late scorer; required AUM is unavailable for most missing cells |
| R10 | P2 | Cleaning taxonomy | Annotation frequency is used to declare unannotated types verified negatives |

**R1 — Rebuild the baseline-comparison contract for the fresh holdout.**

Locations: [V2 §8.1 and §9.3](../v2-plan.md), especially lines 515–520, 701–704,
719–730; [setup](../v2-phase1-setup.md), lines 42–45;
[`utils/v2_preparation.py`](../../utils/v2_preparation.py), lines 64–69.

The current procedure draws a fresh random 30K after merging subsets, but older comparison
instructions still rely on the *old* V1-clean 30K. Using the same new images for V1 and V2
controls paired sampling; it does not equalize training exposure. Illustratively, two
independent 30K draws from the historical 5,921,102-image corpus overlap in only about
**152 images in expectation**. This is an arithmetic example, not a measurement of the
new corpus; genuinely new-source images may provide additional V1-unseen coverage.

The named `run1_vit/checkpoints/last.pt` and `best_model.pt` no longer exist, consistent
with the setup's removal record. V1/V1.1 ONNX and safetensors exports remain. Their
vocabularies use bare names such as `long_hair`; prepared V2 requires category-prefixed
names such as `gen:long_hair`, with a newly measured label universe. A shared scorer
alone does not define the intersection or resolve category collisions.

Before the baseline freeze, specify the surviving export/version and its preprocessing,
pin hashes, create an explicit semantic tag crosswalk, and separate common-label results
from V2-only coverage. Record V1 exposure as verified unseen/seen/unknown where evidence
exists. If the old membership cannot be reconstructed, withdraw the old clean-subset
confirmation promise and report the new comparison's exposure limitation. **No extra
holdout reservation is required by this recommendation.** Training/test exposure can
bias vision comparisons; the magnitude here remains unmeasured.
[Barz & Denzler, ciFAIR](https://pub.inf-cv.uni-jena.de/pdf/barz2020train.pdf).

**R2 — Replace conditional cross-fire with conditional false-positive rate, and protect recall.**

Locations: [V2.1.1 §2, §4, §9](../v2.1.1-gold-training-pipeline.md), lines 42,
63–67, 125–127; [gold taxonomy](../../.research/golden_set_plan.md), lines 83–85,
143–147. Janitor and production Gate 3 explicitly consume this shared metric.

The proposed cell is `P(predicted j = 1 | true i = 1)`. That is not a confusion rate
when labels co-occur. The plan itself specifies `absurdly_long ⇒ very_long ⇒ long`.
A perfect model therefore fires `absurdly_long` on some truly `long` images and fires
`long` on every truly `absurdly_long` image. Color pairs can also both be true, including
multi-character images. Reducing those cells indiscriminately rewards missed positives.
Multi-label classification predicts all applicable labels, rather than requiring one
exclusive answer. [Cole et al., CVPR 2021](https://openaccess.thecvf.com/content/CVPR2021/html/Cole_Multi-Label_Learning_From_Single_Positive_Labels_CVPR_2021_paper.html).

Use `P(predicted j = 1 | reviewed i = 1, reviewed j = 0)` for false-fire cells;
exclude unknown `j`, and mark logically impossible conditioning cells undefined.
Alternatively, define mutually exclusive, entity-specific ordinal bins before building
a multiclass confusion matrix. Keep co-occurrence rates separately named. Require a
paired recall/non-inferiority condition at the declared operating point: a detector
that emits nothing gets zero cross-fire and must not pass the boundary-improvement gate.
The CPU example gives a perfect model 50% proposed cross-fire but 0% conditional false-fire.

**R3 — SEALED must have one role; certification labels cannot fit the certified rule.**

Locations: [V2.1.1](../v2.1.1-gold-training-pipeline.md), lines 52–56, 96,
114, 125; [janitor](../janitor-cleaning-model-plan.md), lines 121, 132,
142–144, 202.

V2.1.1 calls the slice sealed, then requires early stopping/selection on it and temperature
fitting on it. Janitor also allows SEALED results to trigger escalation/recalibration,
and assigns certifying-audit labels to threshold fitting. A once-read slice is no longer
independent of decisions made from that reading. Limiting recalibration to one attempt
does not restore independence. Selection on a finite evaluation sample can itself
overfit, without any gradient step on that sample.
[Cawley & Talbot, JMLR 2010](https://www.jmlr.org/papers/v11/cawley10a.html).

Assign selection to TUNE/development data, calibration to CAL, and final evaluation to
SEALED after all fitted choices are frozen. Audit labels may train a *future* rule, but
the revised rule needs independent certification. If the existing resource budget
requires reuse, call the result development/validation evidence and remove its independent
certification claim. Janitor's intentional exposure to original noisy SEALED images
during Stage A is a separate, explicitly documented in-sample-cleaning estimand; it is
not the leakage finding here.

**R4 — Split before routing disagreements, not after.**

Locations: [janitor §3.1–3.2](../janitor-cleaning-model-plan.md), lines 102,
115–122.

The real slice is intended to contain difficult boundaries, yet **all adjudicated
sidecar–Gemma disagreements go to TUNE**. For that real-slice component, CAL/SEALED
therefore sample conditional on agreement. The cleaner is supposed to fix label errors,
but its qualifying set systematically loses a source of those errors. A random audit of
the remaining agreements can measure shared mistakes; it cannot restore the removed
disagreement population. Other portal data do not establish representative replacement
coverage unless their sampling design explicitly does so.

Freeze image membership before revealing screening outcomes. Route disagreements to
adjudication *within their assigned split*, keeping CAL/SEALED judgments inaccessible
to fitting. Record source, agreement status, strata, and inclusion probabilities.
Importance weighting cannot recover a population stratum with zero evaluation inclusion
probability. The danger of realistic, nonuniform label selection is directly studied
by [Arroyo, Perona & Cole, ICLR 2023 Tiny Papers](https://openreview.net/pdf?id=iWiwox99aJ);
the specific split defect here follows from the plan's routing rule.

**R5 — Use an estimator and bound that match the stratified edit audit.**

Locations: [janitor §6.1–6.2](../janitor-cleaning-model-plan.md), lines 183,
185–186, 199–206.

Fixed quotas per frequency stratum deliberately oversample some candidate populations.
Pooling their successes into one ordinary binomial interval estimates neither edit-volume
precision nor, generally, a binomial experiment. For the former, the point estimate is
`sum_h (N_h / N) * precision_h`, where `N_h` counts actual proposed edits in that stratum.
Its uncertainty must respect the sampling design and any within-image/cluster dependence.
[Penn State Department of Statistics, stratified proportions](https://online.stat.psu.edu/stat506/Lesson06).

Reproduced counterexample: review 100 candidates in each of five strata, obtaining
90, 100, 100, 100, 100 correct. The pooled score is 98%, with a naive one-sided 99%
Clopper–Pearson lower bound of **96.01%**, passing the 95% bar. If the 90%-precision
stratum contains 96% of actual edits, population precision is **90.4%**. This example
isolates weighting and assumes perfect reviewers.

Keep deliberate tail sampling, but predeclare whether the target is volume-weighted
precision, equal-stratum precision, or simultaneous stratum guarantees. Use appropriate
stratified inference or combine simultaneous per-stratum bounds with known weights.
Count the certification unit explicitly: a swap must have a correct removal *and* addition;
two tag operations are not two independent successes.

**R6 — The confidence-bound machinery needs a complete statistical contract.**

Locations: [janitor](../janitor-cleaning-model-plan.md), lines 107, 143,
179–202.

Three distinct omissions remain:

- `p_true = (p_L - alpha) / (1 - alpha - beta)` transforms a confidence endpoint as if
  reviewer error rates were known constants. They are estimated from small audits.
  The transformation needs their joint uncertainty and error rates applicable to the
  *rule's selected candidates and action-correctness event*. Per-axis label error rates
  do not automatically transfer to swaps or high-score candidate subsets.
  [Flor et al., BMC Public Health 2020](https://d-nb.info/1217587721/34) explicitly studies
  coverage when sensitivity and specificity are uncertain.
- Choosing a threshold because its ordinary CAL lower bound clears a bar is still
  data-dependent threshold search. Switching large rules from 95% to 99% is a risk
  preference, not by itself a specified campaign-wide error or bad-edit-volume guarantee.
  Predeclare the estimand, candidate rule family, selection procedure, and error budget,
  or use an untouched audit solely for the already-frozen chosen rules.
  [Angelopoulos et al., Learn then Test](https://arxiv.org/html/2110.01052v5) gives methods
  for valid selection among risk-tested settings; it is an optional methodological
  reference, not a proposal to add a new training arm.
- The example “reject at ≥19 failures” tests evidence of being **below** the 95% bar.
  It is not the acceptance rule for establishing precision **above** that bar. With
  `n=200`, one-sided 99% Clopper–Pearson, and error-free reference labels, certification
  permits **at most 3 failures**: lower bounds are .950652 at 3 and .943071 at 4.
  Counts 4–18 do not certify. The ≥19 example is a valid strong-rejection threshold;
  the missing piece is an explicit inconclusive region, not a reversed binomial calculation.
  [NIST, exact binomial confidence limits](https://www.itl.nist.gov/div898/software/dataplot/refman2/auxillar/exacbici.htm).

Implement a three-way pass/inconclusive/fail outcome and choose either a stated frequentist
coverage method or a stated Bayesian credibility method. Clopper–Pearson and Jeffreys
intervals should not be interchangeable under an “exact” frequentist guarantee. Where
reviewer-error uncertainty is unfunded, report sensitivity bounds or require direct
adjudication rather than certify from a plug-in correction.

**R7 — Correct the binary odds formula and withdraw the known-prior assumption.**

Locations: [V2.1.1 §6](../v2.1.1-gold-training-pipeline.md), lines 94–97;
[janitor §4](../janitor-cleaning-model-plan.md), line 141.

For a calibrated binary posterior under pure label shift, the additive correction is
`logit(pi_target) - logit(pi_train)`, with `logit(p) = log(p / (1-p))`.
The plans use `log(pi_target) - log(pi_train)`, omitting the negative-class odds term.
That log-ratio is appropriate inside normalized multiclass posterior adjustment;
it is not the exact offset for independent one-vs-rest sigmoids. A label-balanced gold
prior makes the missing term consequential. Reproduced example: train prior .5 and
target prior .1; the proposed offset maps an uninformative .5 prediction to **.1667**,
whereas the binary odds correction yields **.1**.

This is a Bayes-rule derivation conditional on unchanged class-conditional feature
distributions. Synthetic and boundary-mined examples need not satisfy that condition.
[Lipton, Wang & Smola, ICML 2018](https://proceedings.mlr.press/v80/lipton18a.html).
Moreover, raw ASL scores are not automatically posteriors, as the main V2 plan already
recognizes. [Cheng & Vasconcelos, CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Cheng_Towards_Calibrated_Multi-label_Deep_Neural_Networks_CVPR_2024_paper.html).

The deployment prior is also not “known” from sidecars. Even with identified missingness
`rho`, `P_sidecar/(1-rho)` requires no wrong positives. If `f` is the false annotation
probability among truly negative images, then
`P_sidecar = (1-rho)*pi_true + f*(1-pi_true)`.
This campaign explicitly targets both omissions and wrong positives. A decile-average
rho additionally needs justification before substituting for a per-tag quantity.
Use the correct odds expression only where its assumptions hold; otherwise treat it as
a sensitivity diagnostic and fit the actual operating point on disjoint, appropriately
sampled reviewed real data. The janitor's initializer-only caveat should apply regardless
of whether cleaning runs before or after V2.1.1.

**R8 — Preserve which labels were actually reviewed through training and evaluation.**

Locations: [janitor](../janitor-cleaning-model-plan.md), lines 103–105, 115–117;
[V2.1.1](../v2.1.1-gold-training-pipeline.md), lines 29–36, 73–76, 125–127;
[`vocabulary.py`](../../vocabulary.py), lines 648–669;
[`dataset_loader.py`](../../dataset_loader.py), lines 2258 and 2474;
[`loss_functions.py`](../../loss_functions.py), lines 275–277, 394–401.

Janitor correctly says a hair-axis synthetic sample is not a verified negative for
unreviewed eye/body tags. That masking rule is scoped to its group refit. V2.1.1 then
promotes these images into a full-corpus training path without specifying how their
observation masks survive. The ordinary vocabulary encoder initializes all absent tags
to zero; the current sample builder explicitly masks unknown **ratings** only. Without
new source-aware encoding, promotion recreates false-negative supervision on every
unannotated rendered attribute. Missing labels and verified absence are different
supervision states. [Cole et al., CVPR 2021](https://openaccess.thecvf.com/content/CVPR2021/html/Cole_Multi-Label_Learning_From_Single_Positive_Labels_CVPR_2021_paper.html).

The same gap affects evaluation: a real slice reviewed for several confusable axes is
not clean ground truth for the other ~19,250 tags. It can measure reviewed-axis behavior;
scoring untouched sidecars measures noisy-label agreement. It cannot certify broad
clean-label non-regression merely because the images themselves are human-reviewed.

Specify per-image positive, verified-negative, and unknown cells, their provenance, and
their serialization through sidecars/Arrow/workers/metrics. Mask unknown cells for these
known partial gold sources; this does not require treating the entire production corpus
as fully unobserved. Keep noisy broad-vocabulary validation separately named and use
reviewed label subsets for clean claims.

Also freeze the resulting weighting: the current loss averages *observed labels per
image*. Reducing an image from ~19K supervised columns to ~13 increases each retained
cell's reduction weight by roughly 1,500× at equal image weight. This may be intentional,
but global synthetic-image percentage alone does not describe its training influence.
Report effective positive/negative exposure per target group and the chosen sample/loss
normalization before promotion.

**R9 — The early checkpoint is currently a filter, not an independent missing-tag proposer.**

Locations: [janitor §2 and §5](../janitor-cleaning-model-plan.md), lines 89–90,
156–166.

Late/EMA scores create M1 candidates first. The early checkpoint scores only those
candidate cells. If training has suppressed a true missing tag below the late admission
threshold/slack, it never reaches the early check. Early evidence cannot rescue a cell
absent from its input set: the resulting proposal set is a subset of the late set.
This matters for the exact training-time suppression concern motivating the design.
[Kim et al., Large Loss Matters, CVPR 2022](https://arxiv.org/abs/2206.03740).

There is a second specification gap: AUM is stored for GT-positive cells plus selected
group columns, but M1 requires AUM for GT-negative candidates across the vocabulary.
Most such cells have no recorded AUM. A required input with no missing-data policy can
silently restrict all-tag M1 to the few instrumented groups.

Define candidate recall separately from candidate precision. If recovery from late
suppression is required, give an early/independent proposal route nonzero coverage beyond
the late set, within an explicit scoring budget; otherwise narrow the claim to reranking
late proposals. Specify a meaningful multi-label margin and AUM coverage/missing policy
for every eligible M1 rule before Stage A, since online history cannot be reconstructed
after training. Measure recall using original noisy labels plus separate adjudicated
truth, retaining undiscovered positives in the denominator.

**R10 — “Usually absent” does not establish verified negatives.**

Locations: [janitor §5](../janitor-cleaning-model-plan.md), line 170;
[gold taxonomy](../../.research/golden_set_plan.md), line 111.

The type-axis rule concludes that unlabeled type is a true negative because 98.5% of
parent-tagged images have no type annotation and a default interpretation is common.
Those observations estimate annotation prevalence, not the probability that an
unannotated subtype is truly absent. This is the same identifiability problem that
V2 §10 correctly recognizes elsewhere. No remaining-omission rate can be established
by counting only observed labels.

Retain the generic M1 proposal/human-queue scope if desired, but describe absent type
annotations as assumed negatives until a reviewed real sample supports a stronger
claim. Do not infer verified-negative status from rarity or implication closure:
complete parent implications among *recorded* subtypes say nothing about unrecorded
subtypes. This finding challenges the inference, not the reported sidecar counts.
[Arroyo, Perona & Cole, label-selection bias](https://openreview.net/pdf?id=iWiwox99aJ).

**Additional evidence corrections with narrower scope.**

- [Gold-plan lines 27 and 192](../../.research/golden_set_plan.md) call inter-annotator
  alpha an accuracy ceiling. Agreement measures reproducibility, not accuracy; reviewers
  can agree on the same error. Keep agreement and adjudicated accuracy separate.
  [Artstein & Poesio, §2.1](https://aclanthology.org/J08-4004.pdf).
- [V2.1.1 line 21](../v2.1.1-gold-training-pipeline.md) turns the feature-distortion
  paper into an unconditional fine-tuning prohibition. Its findings depend on good
  pretrained features and distribution shift; it also finds benefits from linear probing
  followed by fine-tuning. Keep the chosen from-scratch/frozen-backbone scope as a project
  decision without presenting the paper as a universal impossibility result.
  [Kumar et al., ICLR 2022](https://arxiv.org/abs/2202.10054).
- [Janitor line 225](../janitor-cleaning-model-plan.md) says lower missingness strengthens
  the case for clip=.1. Increasing the ASL probability shift from .05 to .1 suppresses
  negative learning more. Lower missingness alone does not imply that change; the main
  plan's own §4 derivative already establishes the direction.
  [ASL, §2.3](https://arxiv.org/html/2009.14119v4).
- [V2 §6 lines 398–403](../v2-plan.md) still treats the historical 512px corpus geometry
  as the unconditional native terminus. The added subset's geometry is unmeasured.
  Keep the adopted resolution schedule, but recount native dimensions, aspect ratios,
  and source-specific small-detail performance before claiming 512 is native for the
  combined corpus. The same source-stratified audit should check label coverage: a
  mixed-source aggregate can hide source-specific failures. This is a transferability
  question, not evidence that a higher resolution will win.

**What the evidence supports keeping.**

The corrected main plan accurately distinguishes SoViT's 880×18/MLP2320/patch14 shape
from the local 896-wide adaptation, and distinguishes the usual ASL 4/0/.05 experiment
from the OpenImages gamma-7 recipe with reduced-weight assumed negatives.
[SoViT §4.1](https://arxiv.org/html/2305.13035v5),
[ASL Appendix F](https://arxiv.org/html/2009.14119v4).
The constant-LR-plus-cooldown literature supports WSD as a practical schedule candidate;
it does not establish this dataset's optimal plateau threshold or convergence budget.
[Hägele et al., NeurIPS 2024](https://arxiv.org/html/2405.18392v3).
These checks do not justify replacing the selected backbone, changing gamma, adding
pretraining, or launching new training arms. Exact-versus-binned AP checks, unknown-rating
masking, the finite budget/cooldown distinction, and the main plan's explicit limits on
noisy-label validation already address earlier findings and are not re-reported as gaps.

The documented upstream deduplication contract and the deliberate 30K-only reservation
are accepted. No new image-hashing gate, additional holdout, production preparation,
or full-model run was performed for this review. Existing augmentation D1–D4 remain
pending decisions rather than newly discovered defects or silently adopted changes.

**Verification and recommended order.**

CPU-only reproductions passed for exact-binomial acceptance counts, the stratified-audit
counterexample, binary prior odds, legitimate hierarchical co-fire, and the early-candidate
filter. The local probe is
[`.code-review/v2_plan_scholarly_checks_20260906.py`](../../.code-review/v2_plan_scholarly_checks_20260906.py)
(git-ignored review artifact). Its numerical examples are fully specified above; run with
`L:\Dab\payton_env\Scripts\python.exe -B .code-review/v2_plan_scholarly_checks_20260906.py`.
No new model-quality measurements or production performance claims are made.

Before the production comparison freeze, resolve R1 and name the comparison's supported
claims. Before collecting/splitting the janitor's real slice, resolve R2–R4 and R8's
annotation scope; those choices determine what review work is reusable. Before Stage A,
resolve R9's online-history requirements. Before any automatic cleaning, implement and
validate R5–R7 and the corrected R10 assumptions. Amend the owning plans after decisions;
this evidence report does not itself approve those implementation changes.
