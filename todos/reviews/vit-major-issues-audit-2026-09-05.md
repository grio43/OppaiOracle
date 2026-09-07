# ViT major-issues audit — 2026-09-05

Scope: the current working tree in `L:\Dab\OppaiOracle`, its canonical YAML, and the **current** [V2 plan](../v2-plan.md). The open pre-rewrite archive is historical evidence, not the specification. The original audit below made no production edits; subsequent remediation is tracked separately in the status section. ONNX/export, unused CLI paths, minor cleanup, and speculative architecture improvements are excluded.

**The most consequential verified problems are loss-gradient saturation under bf16, holdout contamination when validation is capped, and checkpoint paths that can discard or replace the intended resume state.** V2 also has explicitly planned launch blockers that remain implemented according to the old policy. They are identified as such below; an unfinished migration is not presented as an unexplained regression.

## Remediation status

The original reproductions, source line numbers, and fingerprints below describe the **pre-fix**
working tree. They are retained as evidence, not as descriptions of current behavior.
Only the training pipeline was changed. Existing unrelated working-tree edits were preserved.

| ID | Current implementation | Verification / remaining scope |
|---|---|---|
| 1 | Probability, logarithm and focal math use fp32 before bf16 saturation; focal weights detach | CPU/CUDA bf16 gradients match the independent detached fp32 reference at logits 0, 2, 6.25 and 7, clips .05/.2 |
| 2 | Trainer casts logits before sigmoid; diagnostics preserve fp32 probabilities | Actual trainer expression yields AP 1.0 on the original counterexample. Pre-fix best/burn-in/stop history resets on resume; a frozen production-subset comparison remains unmeasured |
| 3 | User revised the requirement: cap is the only holdout (30K); excess candidates return to training | The extra reservation fix is withdrawn at user request. Tests check capped holdout disjointness and excess-row coverage through real loaders and worker reopening; larger CALIB/TEST sets are deferred |
| 4 | Found incompatible `latest`/`best` raises instead of starting fresh | Both automatic modes and explicit mode tested |
| 5 | Final completed epoch and policy halt save current state even when not best | Real tiny CUDA trainer, ordinary completion during burn-in and non-best policy halt; final optimizer state, epoch boundary and evaluation history checked |
| 6 | Synchronous fallback waits for earlier submissions; callbacks update pointers before retention | Deterministic queue saturation tested for normal and best-only retention; newest last and eligible best survive; write errors propagate at shutdown |
| 7 | Fixed ASL tuple checked after resume; conflicting checkpoint gamma aborts; override/schedule refused | Config and live criterion checked, including telemetry disabled; gamma-step API has no authority |
| 8 | Holding/rising sibling gap never demotes a decline; falling gap may corroborate | Updated production stop replays and existing stop suite |
| 9 | Parent and worker access the full mmap through selected row indices, with no `take()` payload copy | Disjoint even/odd selections survive pickle/reopen; 10K-row test allocates under 64 KiB of Arrow payload on first worker read |
| 10 | One fp32/bool host buffer; bucket scratch bounded to 32 columns; exact AP computed per label | Chunked values match independent TorchMetrics reference, including zero-support and singleton buckets. Full EVAL/persistent-worker peak RSS and duration remain unmeasured |

Verification commands: `python -B test_training_audit.py`, `python -B test_stop_conditions.py`,
`python -B test_softstop_resume.py`, `python -B test_softstop_resume_e2e.py`,
`python -B test_flip_pipeline.py`,
`python -B test_rotation_mask.py`, and canonical config validation. All training fixtures use
temporary data/checkpoints. No production training, checkpoint rewrite, or corpus mutation was run. The existing resume fixture
was updated to disable stop-window coverage for its two-epoch run and to match the already-fixed
scheduler constructor position (0, rather than the old 1); it now checks resumed loss settings too.

Verification before the user revised the holdout budget: **11/11 audit tests, 46/46 real-process resume checks, 39/39 stop-policy
checks, 33/33 structural/component resume checks**, all flip-pipeline checks, all six
rotation-mask groups, canonical config validation, and `git diff --check` passed.

After the user restored the 30K-only holdout, the cap regression and real CUDA
completion/halt fixture passed again: 196 training / 4 validation rows out of 200,
including worker reopening and no overlap. Flip-worker checks, config validation,
and `git diff --check` also passed. The extra holdout reservation is intentionally withdrawn.

At the current EVAL size 30,000 × 19,294, the retained fp32/bool host buffer is **2.89 decimal GB**. The former 146,056-row proposal would use 14.09 GB; that larger reservation is deferred.
There is no concatenation copy or full int64 target matrix in frequency diagnostics. Additional
scratch is bounded by 32 columns for buckets and row chunks for ASL telemetry. This is allocation
accounting, not a full production RSS measurement. Headline AP remains the 200-bin estimator;
bucket AP remains exact. TensorBoard off no longer suppresses enabled ASL validation telemetry.

### Additional major training findings encountered

- **Missing explicit resume path started fresh — fixed.** A typo/nonexistent checkpoint now raises
  `FileNotFoundError`; it cannot silently start a new model in the requested output directory.
- **Async save failures could end successfully — fixed.** Shutdown now raises on pending-save
  timeout or writer failure, pointer update failures propagate, and trainer cleanup preserves the
  error for its caller. Retention runs after pointer callbacks, including under best-only saves.
- **ASL validation telemetry depended on TensorBoard — fixed.** Its enable flag now independently
  activates accumulation. The corrected buffer also removes the previous fp16 rounding of
  diagnostic scores, which could disagree with the corrected headline metric.
- **Single-tag/zero-support frequency buckets could fail — fixed.** The old multilabel helper
  required at least two columns; bounded per-label AP/count reductions handle these buckets.
- **Configured scheduler can still be ignored — open, already identified by this audit.** The
  trainer constructs cosine directly despite accepting other schema values. WSD plus trainer
  dispatch remains in V2 §12; changing the scheduler name alone is insufficient.

These fixes do **not** claim full V2 launch readiness. Architecture/WSD migration, actual split fingerprinting,
full-scale memory validation, and the other §12 gates remain open.

## Audit basis and verification limits

- Git HEAD: `ad17ee98efeaa3314a0b2fb73499cd81c4099eb0`. The tree already contained modified and untracked implementation files; findings describe those working files, not HEAD alone. Source fingerprints are at the end.
- Interpreter: `L:\Dab\payton_env\Scripts\python.exe`; PyTorch `2.11.0+cu130`, TorchMetrics `1.8.2`, PyArrow `22.0.0`; RTX 5090; **255.11 GiB host RAM**. The YAML's 128 GB-era comments do not describe the installed RAM.
- Actual vocabulary: **19,294 columns**, including PAD, UNK, and four rating tags. The metadata IPC contains **5,921,102 rows**, with `table.nbytes = 4,156,804,141`.
- Verification used source tracing, production functions executed on bounded synthetic inputs, a small real CUDA model, temporary checkpoint files, and a 10,000-row sample of the real metadata cache. The production dataset, vocabulary, and training checkpoints were not modified.
- A synthetic counterexample proves a reachable failure, not its prevalence in a previous run. No full-corpus mAP drift, production OOM, or historical checkpoint loss is claimed here.
- P1 means training correctness, evaluation integrity, or checkpoint preservation should be addressed before a long affected run. P2 means a material resource cost that needs explicit budgeting; it does **not** mean this 255 GiB machine necessarily runs out of memory.

## Findings at a glance

| ID | Priority | Issue | Verification | V2 relationship |
|---|---|---|---|---|
| 1 | P1 | bf16 sigmoid kills the negative-label gradient for sufficiently confident predictions | Real loss, CUDA autocast, fp32 comparison | Persists after changing clip or detaching focal weights |
| 2 | P1 | Validation casts after sigmoid, losing distinctions used by model selection | Actual 200-bin AP counterexample | Affects the selection instrument |
| 3 | P1 | Validation capping puts reserved holdout images into training | Executed production cap block | Known V2 split blocker, still present |
| 4 | P1 | `latest`/`best` can turn an incompatible resume into fresh training | Executed production catch block and real compatibility validator | Known V2 fail-fast blocker |
| 5 | P1 | Ordinary completion has no unconditional final checkpoint | Real two-epoch trainer run plus all save/exit paths traced | Relevant to phase completion/cooldown |
| 6 | P1, conditional | A full async queue can roll `last.pt` and `best_model.pt` backwards | Real manager/writer with deterministic backlog | Independent of the V2 recipe |
| 7 | P1, V2 blocker | The fixed, detached V2 loss is not enforced | Gradient comparison and real loss-state reconciliation | Deliberate migration still outstanding |
| 8 | P1, V2 blocker | Rising sibling margin still demotes a genuine stop trigger | Paired production stop-policy replay | Explicitly withdrawn V2 behavior |
| 9 | P2 | Each worker copies its selected Arrow table into private memory | Real accessor pickle/reopen, allocator measurement | About 22 GiB for six train workers at the intended split |
| 10 | P2 | Validation diagnostics expand well beyond compact host buffers | Exact dtype/shape accounting and consumer tracing | EVAL expansion needs a larger budget than concatenation alone |

## 1. bf16 saturation removes the negative-label learning signal

**Locations:** [loss_functions.py](../../loss_functions.py), lines 278–335; [train_direct.py](../../train_direct.py), lines 2162–2189; model output at [model_architecture.py](../../model_architecture.py), lines 702–722.

The model's tag-head output is bf16 under the production autocast policy. The loss computes `sigmoid(logits)` without first converting logits to fp32. Its negative term, including the shifted probability and focal weight, depends on that sigmoid. Once it rounds to exactly 1, sigmoid backward returns zero.

**Verified on CUDA:** for a negative target, `gamma_neg=7`, `gamma_pos=0`, `alpha=1`, no smoothing, and unreduced per-cell loss:

| clip | logit | bf16 sigmoid | bf16-path gradient | fp32-input gradient |
|---|---:|---:|---:|---:|
| 0.20, current | 6.25 | 1.0 | **0** | 0.00752834 |
| 0.20, current | 7.0 | 1.0 | **0** | 0.00360468 |
| 0.05, V2 | 6.25 | 1.0 | **0** | 0.05441289 |
| 0.05, V2 | 7.0 | 1.0 | **0** | 0.02626664 |

The loss is nonzero and finite at these points, so NaN/Inf guards do not detect this. The affected prediction may still move through shared features, other samples, optimizer momentum, or weight decay; the precise failure is that **this negative example supplies no corrective gradient**. Its frequency in real checkpoints was not measured.

**Counterchecks:** the actual tiny ViT produced bf16 logits; the same loss with fp32 inputs had nonzero gradients; both the current and planned clip values reproduce it. Positive labels use stable BCE-with-logits separately, so this is not a claim that every positive gradient disappears. The returned loss was already fp32 under autocast: changing only the final reduction dtype does not repair the earlier saturation. PyTorch's [autocast reference](https://docs.pytorch.org/docs/stable/amp.html) explains the per-operation casting policy; the installed-library reproduction is the decisive evidence here.

**Closure criterion:** test confidently wrong negatives with the complete intended V2 loss under CUDA bf16 autocast, ensuring probability/log/focal computations preserve the fp32 reference gradient before backward casts back to bf16. Detach and precision are separate changes.

## 2. The selection metric loses score distinctions before its fp32 cast

**Location:** [train_direct.py](../../train_direct.py), line 3039: `torch.sigmoid(outputs['tag_logits']).float()`; lines 3044–3048 feed these values to the selection metrics. CPU diagnostics inherit the same values at lines 3084–3085.

Casting the result cannot recover information lost in bf16 sigmoid. This is distinct from the unavoidable quantization of the model's bf16 logits: two **different representable bf16 logits** can become an identical probability, including across one of the configured AP bins.

**Verified with the actual TorchMetrics 200-bin estimator:** two examples, each repeated across two label columns, with the lower score negative and the higher score positive:

| | Current expression | Cast logits before sigmoid |
|---|---|---|
| Input logits | `[-1.0703125, -1.0625]` | Same |
| Probabilities | `[0.255859375, 0.255859375]` | `[0.25534365, 0.25683200]` |
| Binned mAP | **0.5** | **1.0** |

This is a counterexample, **not a claim of a 0.5 corpus-wide error**. It establishes that the extra rounding changes the instrument used to select checkpoints and detect plateaus. It can also change threshold decisions near a boundary. The training-side ASL telemetry already casts logits before sigmoid (`asl_telemetry.py:448`), so the two paths do not share a precision contract.

**Closure criterion:** standardize the score computation and quantify the change on a frozen validation subset. Preserve V2's same-estimator comparison rule; old and corrected metric series need an explicit compatibility check. Raising `ap_thresholds` does not recover distinctions already lost before binning.

## 3. Setting the V2 validation size still trains on CALIB/TEST's reserved pool

**Locations:** [dataset_loader.py](../../dataset_loader.py), lines 2782–2808; [V2 plan §8.1](../v2-plan.md#81-split-budget--decided-2026-07-30).

After the 95/5 split, the cap takes `val_list[:max_val_samples]` and appends the excess to `train_list`. This is the primary sidecar-loader path, not a CLI-only behavior.

**Verified:** executed the exact production `if max_val_samples ...` AST block against 296,056 synthetic holdout IDs and `max_val_samples=146056`. Result: **146,056 validation IDs and 150,000 formerly reserved IDs appended to training**. With the current 30,000 cap, the corresponding fold is 266,056 IDs. The original train and validation lists remain disjoint after the operation; the defect is that the broader reserved pool is no longer held out.

Changing the cap alone therefore does not implement V2's EVAL/CALIB/TEST separation. The loader has no separate CALIB/TEST exclusion mechanism in this path. Reopening workers preserves the selected rows, so workers do not undo this contamination.

**Closure criterion:** freeze and hash the intended sets, prove train/EVAL/CALIB/TEST disjointness at the IDs actually served by workers, and keep unused validation images outside training. This is already a V2 launch blocker; the audit confirms it remains live.

## 4. An incompatible automatic resume can start fresh in the same experiment

**Locations:** [train_direct.py](../../train_direct.py), lines 1091–1120; [training_utils.py](../../training_utils.py), lines 472–477 and 598–608.

The pre-load validator raises on a critical mismatch. For `resume_from: latest` or `best`, the trainer catches that exception, logs a warning, sets `ckpt_path=None`, and continues with the freshly initialized model/optimizer. The current YAML uses `latest`.

**Verified:** used the real compatibility validator with current vocabulary size 19,294 and a synthetic checkpoint preview containing 19,293. Executing the production catch block produced:

```text
latest      -> ckpt_path=None, no exception
best        -> ckpt_path=None, no exception
explicit.pt -> ValueError
```

This is not completely silent in the logs, but it is operationally a successful launch that has discarded the requested continuation. The output directory remains the same, so subsequent checkpoints can replace previous run pointers and retention can remove older numbered files.

**Counterchecks:** a same-size vocabulary SHA mismatch is a different guard; this reproduction does not imply that all vocabulary mismatches are ignored. The image-resolution transition was deliberately removed from the critical list and is not the trigger used here. Ordinary, compatible resume tests pass.

**Closure criterion:** a found but critically incompatible automatic-resume checkpoint must abort with its diff. Fresh training should be an explicit decision. Already required by V2 §12.

## 5. Finishing normally does not guarantee that the final state is saved

**Locations:** [train_direct.py](../../train_direct.py), periodic save at 2419, soft-stop saves at 2533/2852, one-shot save at 2601, best save at 3504–3537, finalization at 3563–3644.

There are exactly five checkpoint-save call sites. At epoch end the only ordinary save is inside `if is_best`. After the epoch loop the trainer closes the writer and monitor but does not save the current state. `save_best_only: false` does not add a final or every-epoch save.

**Verified by control-flow inspection:** all save call sites have periodic, stop, manual-save, or best-model conditions. Thus a non-best final epoch that ends between periodic saves returns without persisting its final weights, optimizer state, and completed validation history. Likewise a halt on a non-best validation does not checkpoint that verdict before breaking. The current cadence is every **10,000 optimizer updates**, so this can lose a substantial fraction of the final epoch from the resume artifact.

**Also verified through the real trainer:** reused the existing end-to-end fixture's 200 synthetic images and tiny two-layer model, with all output/cache paths redirected to a temporary directory. Ran two epochs with the current two-epoch burn-in, validation every epoch, `save_steps=10000`, and `save_best_only=False`. The function returned normally and the checkpoint directory contained **zero `.pt` files**. Stop mode was disabled in memory so the intentionally short fixture satisfied the current minimum stop-window validation; AdamW replaced the quantized optimizer only for this small reproduction. No training/checkpoint logic was mocked. This demonstrates the missing completion guarantee; it is not a claim that the 15-epoch production recipe ordinarily writes no checkpoints at all.

The best checkpoint can correctly remain the best; the missing guarantee concerns `last.pt` and final training state. The async shutdown merely drains already-submitted saves and cannot create the missing one.

**Closure criterion:** on ordinary completion and policy halt, verify that `last.pt` records the final committed update, completed-epoch status, and latest validation history, including when the final model is not best. This matters for phase transitions and any future cooldown branch.

## 6. Async queue saturation can overwrite newer checkpoint pointers with older ones

**Locations:** [training_utils.py](../../training_utils.py), lines 1101–1136, 1559–1647, 1658–1680.

When `save_async()` reports a full queue, `CheckpointManager` writes the newest save synchronously and refreshes `last.pt`/`best_model.pt`. Older queued saves can complete afterward, and their callbacks unconditionally refresh the same pointers. File locking and atomic replacement protect file integrity, but do not enforce chronological ordering or best-score ordering.

**Verified with the real manager and writer:** temporarily delayed the worker's start, leaving the normal writer implementation unchanged once released. A queue of capacity two accepted steps 1 and 2; step 3 fell back to synchronous saving. All three were increasing best scores. Results:

```text
After synchronous fallback: last.pt step = 3
After older queued saves drain: last.pt step = 2; best_model.pt step = 2
```

All files were in a temporary directory. Ten numbered checkpoints were allowed, so retention deletion was not necessary to reproduce the failure. No production checkpoint was opened or overwritten.

**Trigger/limit:** a real backlog is necessary, such as slow storage plus closely spaced periodic/manual/epoch saves. This is not evidence that the present 10,000-update cadence has encountered it. The potential loss is nevertheless major because default resume prefers `last.pt` and the best pointer can stop identifying the best saved model.

**Closure criterion:** force queue saturation and completion reordering, then verify that the latest pointer never regresses and the best pointer still identifies the highest eligible score.

## 7. The effective loss still follows the old policy, despite V2's fixed-loss decision

**Locations:** [loss_functions.py](../../loss_functions.py), lines 307–335; [asl_telemetry.py](../../asl_telemetry.py), lines 136–190; [configs/unified_config.yaml](../../configs/unified_config.yaml), lines 380–428; [V2 plan §4](../v2-plan.md#4-loss--asl-cited-correctly) and §12.

Two independently verified channels remain:

1. **Focal weights remain differentiable.** For negative labels, fp32 inputs, clip 0.05 and gamma 7, the live gradients at logits `[0, 2]` were `[0.01038602, 0.59889162]`. Detaching the focal weight while retaining the same forward loss gave `[0.00169850, 0.16951942]`. This is a different gradient objective, not just a configuration label or precision effect.
2. **Checkpoint gamma overrides YAML.** Constructing the real `ASLDriveManager` with YAML gamma 7 and persisted `loss_state.gamma_neg=4` changed the live criterion to **4**, with a warning. Turning telemetry off did not prevent reconciliation. A null manual override does not prevent it either.

The current YAML also retains clip 0.2 and an enabled guarded gamma-step policy. It does **not** automatically descend gamma just because the policy is enabled; the confirmed problems are persisted-state precedence, retained mutation authority, and lack of the V2 conformance check.

**Classification:** known, planned V2 changes that are still required before a V2/janitor run. Non-detach is not retrospectively called a V1 implementation bug. Merely changing clip to 0.05 or writing gamma 7 in YAML does not land the V2 loss. Issue 1 remains even after detaching.

**Closure criterion:** verify the effective criterion after all checkpoint reconciliation, including a deliberately conflicting persisted loss state; enforce the intended fixed-loss contract before training begins.

## 8. The stop system still trusts rising sibling margin as permission to continue

**Locations:** [stop_conditions.py](../../stop_conditions.py), lines 784–810; [test_stop_conditions.py](../../test_stop_conditions.py), lines 254–274; [V2 plan §9.2](../v2-plan.md#92-sibling-negative-evaluation--veto-only-never-a-green-light).

V2 explicitly rejects this inference: memorizing a wrong positive can increase the sibling gap, so a rising gap must not demote an otherwise valid decline trigger. The implementation still marks `peak_decline.advisory=True` in that case.

**Verified with paired production evaluations:** mAP history `[.60, .65, .70, .72, .70, .69]`, the same policy (`halt`, regression confirmation 2), same losses and epoch count:

| Sibling margins | Decline tripped | Advisory | Overall action |
|---|---|---|---|
| Missing | Yes | No | `halt` |
| `[.10, .12, .14, .16, .18, .20]` | Yes | **Yes** | **`continue`** |

The existing stop suite passes **39/39**, including a test that asserts this withdrawn behavior. Passing that suite therefore does not establish V2 conformance. The current YAML uses `advise`, so this does not secretly cause an automatic halt today; it still downgrades the operator's stop signal, and becomes a mechanical continuation bug in halt mode.

**Closure criterion:** a holding/rising gap must leave the underlying decline trigger intact; a falling gap may corroborate it. Update the behavioral expectation as well as the implementation when this planned change is made.

## 9. Worker reload materializes private Arrow tables despite mmap

**Locations:** [dataset_loader.py](../../dataset_loader.py), lines 875–906, 2005–2015, and 2091–2114.

The parent selects its rows using `arrow_table.take(row_indices)`. Each spawned worker reopens the full mmap, then repeats `table.take(self._row_indices)`. This preserves split correctness, but the selected variable-length metadata buffers are allocated anew in every worker. Clearing `json_files` from the pickle does not remove those allocations.

**Verified using real cache rows and `ArrowMetadataAccessor`:** copied 10,000 rows into a temporary IPC, selected 9,500, pickled/unpickled the accessor, and read one row to trigger lazy loading. PyArrow's allocator grew by **6,679,936 bytes**, closely matching the selected table's **6,679,550 bytes**. Parent and restored-worker row contents matched. Opening the mmap alone was not the dominant allocation.

Using the full cache's actual size, six training workers each materializing roughly 95% of its metadata implies **22.07 GiB of private Arrow payload**, before parent tables, row-index arrays, Python/runtime overhead, pinned batches, validation workers, and checkpoints. This is a proportional estimate, not a measured full-process RSS peak. The present smaller holdout leaves an even larger training fraction.

The machine has enough RAM that this is not an established OOM. It is a substantial per-worker scaling cost hidden by the sharing description. Apache Arrow documents [zero-copy slicing separately from row selection](https://arrow.apache.org/docs/python/generated/pyarrow.Table.html); allocation behavior here was also measured on the installed version.

**Closure criterion:** maintain the split/exclusion mapping while verifying worker-private memory does not grow by a selected-table copy per worker. Simply removing the row selection would reintroduce train/validation leakage.

## 10. Validation diagnostics have a much larger host-memory footprint than their compact buffers

**Locations:** [train_direct.py](../../train_direct.py), lines 2984, 3084–3085, 3195–3214; [evaluation_metrics.py](../../evaluation_metrics.py), lines 447–493.

With the current `training.use_tensorboard: true`, every validation probability and target is retained on CPU. The buffers are compact, but the consumers are not: concatenation upcasts probabilities to fp32, and `FrequencyBucketMetrics` converts **the entire target matrix** to int64 before selecting a bucket. Bucket selection then creates additional dense copies. Bucket mAP is exact, with a per-label precision/recall computation in installed TorchMetrics, even though the headline metric uses 200 bins.

**Verified dtype/shape accounting, decimal GB, all 19,294 columns:**

| Live allocation stage | Current 30,000 rows | V2 EVAL 146,056 rows |
|---|---:|---:|
| Retained fp16 probabilities + bool targets | 1.74 | 8.45 |
| Concatenated fp32 probabilities + bool targets | 2.89 | 14.09 |
| Concatenation expression peak, including original lists and fp16 intermediate | 5.21 | 25.36 |
| Additional full int64 target matrix in bucket consumer | **4.63** | **22.54** |

At V2 size, that bucket consumer starts with roughly **36.63 GB** across the concatenated inputs and full int64 target copy, **before** bucket-specific copies and metric scratch space. The largest configured bucket contains 7,873 columns; just one fp32-probability/int64-target pair for it adds approximately **13.8 GB**. Support-filtered copies, when required, and F1/AP work must be included in an actual peak measurement.

These stages are **not all simultaneous**; do not add every table row. This is verified allocation accounting, not a measured end-to-end peak or runtime prediction. In particular, the exact AP implementation iterates over labels: this report does not invent a full `N×L×N` sorting allocation. The 255 GiB host may accommodate it, but the V2 plan's compact-buffer/concatenation allowance alone understates the live consumer budget. The worker copies in issue 9 add concurrent memory pressure.

**Closure criterion:** measure the entire validation pass, including bucket consumers and persistent workers, at the intended EVAL size. Bound/chunk diagnostics or deliberately reserve the measured budget. Turning TensorBoard off alone also removes intended val telemetry when calibration is disabled, so it is not a feature-equivalent solution.

## Other V2 readiness facts — planned work, not additional surprise bugs

| Required V2 behavior | Verified working-tree behavior | Evidence |
|---|---|---|
| WSD: warmup, constant body, explicit cooldown | Trainer directly constructs `CosineAnnealingWarmupRestarts`; WSD is not accepted by schema | `train_direct.py:847–856`; `Configuration_System.py:1625–1627` |
| Single canonical scheduler choice actually controls training | Trainer construction does not dispatch on `training.scheduler` | Same construction site; setting a different accepted scheduler name alone would still construct cosine |
| Option C, width 896 / 18 layers / MLP 2320, LayerScale, QK-norm | Current YAML retains width 1024 / 18 / 4096; current block has pre-LayerNorm, QKV and MLP, without the planned LayerScale/QK-norm | YAML model block; `model_architecture.py:226–303` |
| Launch/resume V2 conformance assertion | Current YAML loads successfully with cosine and clip 0.2; checkpoint gamma can override it | Actual config load and issue 7 reproduction |

These are consistent with the current plan's warning that implementation has not landed. The audit does not recommend substituting an unplanned model or loss. A configuration-only rename of the run is insufficient to launch V2.

## Checks that passed / concerns not promoted

- `python -B test_stop_conditions.py`: **39/39 passed**, with the policy caveat in issue 8.
- `python -B test_softstop_resume.py`: **33/33 passed**. These are the existing structural/component checks, not the complete multi-process end-to-end suite.
- `python -B test_flip_pipeline.py`: all checks passed, including pixels, distribution, and epoch updates reaching already-spawned persistent workers. The old frozen-worker-epoch bug is not current.
- `python -B test_rotation_mask.py`: all six groups passed, including independent geometry and actual dataset loading. The working-tree rotation mask fix is present.
- Tiny CUDA ViT: two layers, width 64, two heads, 32-pixel input, partial padding mask, bf16 autocast; finite outputs and gradients. With dropout disabled, checkpointing on/off produced exactly matching outputs and parameter gradients. No checkpoint-closure or masked-attention failure was reproduced by this check; it does not validate production-scale compilation or all shapes.
- Validation positive-support counts are reset (`train_direct.py:3177`); they are not accidentally cumulative across epochs.
- Checkpoint tensor snapshots are cloned onto CPU before async submission. The queue-order failure is not a mutable-tensor snapshot race.
- Existing binned AP transient memory is already explicitly budgeted in the code/config. The configured 200 bins are not themselves reported as a newly discovered bug. No production-scale GPU memory stress test was run.
- The unused `FullConfig.compute_effective_batch_size()` defect, vestigial orientation files, and legacy CLI/manifest conveniences were not promoted into major primary-path findings.

## Small reproduction examples

Run snippets from the repository using the project interpreter, for example a PowerShell here-string piped to `L:\Dab\payton_env\Scripts\python.exe -B -`. These examples do not start production training.

### Loss precision and selection precision (issues 1–2)

```python
import torch
from loss_functions import AsymmetricFocalLoss
from torchmetrics.functional.classification import multilabel_average_precision

for clip in (0.2, 0.05):
    for dtype in (torch.bfloat16, torch.float32):
        x = torch.tensor([[6.25, 7.0]], device='cuda', dtype=dtype,
                         requires_grad=True)
        fn = AsymmetricFocalLoss(gamma_pos=0, gamma_neg=7, alpha=1,
             clip=clip, label_smoothing=0, ignore_indices=[], reduction='sum')
        with torch.autocast('cuda', dtype=torch.bfloat16):
            loss = fn(x, torch.zeros_like(x))
        loss.backward()
        print(clip, dtype, loss.dtype, x.grad.float().tolist())

x = torch.tensor([[-1.0703125]*2, [-1.0625]*2], dtype=torch.bfloat16)
y = torch.tensor([[0, 0], [1, 1]])
for p in (x.sigmoid().float(), x.float().sigmoid()):
    print(multilabel_average_precision(p, y, num_labels=2, thresholds=200))
# AP: tensor(0.5000), tensor(1.)
```

### Queue rollback (issue 6)

```python
import tempfile, threading, torch
from pathlib import Path
from Configuration_System import load_config
from training_utils import CheckpointManager, AsyncCheckpointWriter, TrainingState

gate = threading.Event()
class DelayedWriter(AsyncCheckpointWriter):
    def _worker(self):
        gate.wait()          # Deterministically model a backlogged writer.
        super()._worker()   # Actual production queue, saves, and callbacks.

cfg = load_config('configs/unified_config.yaml')
with tempfile.TemporaryDirectory(prefix='oo_ckpt_audit_') as td:
    root = Path(td)
    manager = CheckpointManager(root, keep_best=False, max_checkpoints=10,
                                async_save=False)
    manager._async_writer = DelayedWriter(max_queue_size=2)
    model = torch.nn.Linear(1, 1)
    opt = torch.optim.SGD(model.parameters(), lr=.1)
    try:
        for step in (1, 2, 3):
            manager.save_checkpoint(model, opt, None, 1, step,
                {'mAP': step/10}, TrainingState(), is_best=True,
                config=cfg.to_dict())
        print(torch.load(root/'last.pt', weights_only=False)['step'])  # 3
    finally:
        gate.set()
        manager.shutdown(timeout=20)
    print(torch.load(root/'last.pt', weights_only=False)['step'])      # 2
    print(torch.load(root/'best_model.pt', weights_only=False)['step'])# 2
```

### Loss policy and stop policy (issues 7–8)

```python
import torch
from Configuration_System import load_config
from loss_functions import AsymmetricFocalLoss, MultiTaskLoss
from asl_telemetry import ASLDriveManager
import stop_conditions as sc
from test_stop_conditions import _history, _policy

x = torch.tensor([[0., 2.]], requires_grad=True)
loss = AsymmetricFocalLoss(gamma_pos=0, gamma_neg=7, alpha=1,
    clip=.05, label_smoothing=0, ignore_indices=[], reduction='sum')
loss(x, torch.zeros_like(x)).backward()
print('live gradient', x.grad)
z = x.detach().clone().requires_grad_()
p = z.sigmoid()
ref = -((p-.05).clamp_min(0).pow(7).detach()
        * (1-p+.05).clamp_max(1).log()).sum()
ref.backward()
print('detached gradient', z.grad)

cfg = load_config('configs/unified_config.yaml')
cfg.training.asl_telemetry.enabled = False
criterion = MultiTaskLoss(AsymmetricFocalLoss(gamma_neg=7, alpha=1, clip=.05))
drive = ASLDriveManager(cfg, criterion, None, torch.device('cpu'),
    {'gamma_neg': 4., 'phase': cfg.training.phase}, 0)
print('effective gamma', criterion.gamma_neg)  # 4.0, despite YAML 7.0

for clean in (None, [.10, .12, .14, .16, .18, .20]):
    verdict = sc.evaluate(_history([.60, .65, .70, .72, .70, .69],
        clean_margins=clean), _policy(), completed_epochs=6)
    print(verdict.action)  # halt, then continue
```

### Split and resume control flow (issues 3–4)

These execute the relevant production blocks through Python's AST, without starting a trainer or reading a real checkpoint.

```python
import ast, logging
from pathlib import Path
from Configuration_System import load_config
from training_utils import validate_config_compatibility

tree = ast.parse(Path('dataset_loader.py').read_text(encoding='utf-8'))
fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
          and n.name == 'create_dataloaders')
cap = next(n for n in ast.walk(fn) if isinstance(n, ast.If)
           and ast.unparse(n.test).startswith('max_val_samples and'))
scope = dict(max_val_samples=146056, val_list=list(range(296056)),
             train_list=['original_train'], logger=logging.getLogger('audit'))
exec(compile(ast.Module(body=[cap], type_ignores=[]), 'dataset_loader.py', 'exec'), scope)
print(len(scope['val_list']), len(scope['train_list'])-1)  # 146056, 150000

tree = ast.parse(Path('train_direct.py').read_text(encoding='utf-8'))
guard = next(n for n in ast.walk(tree) if isinstance(n, ast.Try)
    and any(h.type and ast.unparse(h.type) == 'ValueError'
            and 'Skipping incompatible checkpoint' in ast.unparse(h)
            for h in n.handlers))
cfg = load_config('configs/unified_config.yaml')
cfg.model.num_labels = 19294  # The trainer resolves num_labels before this guard.
preview = cfg.to_dict()
preview['model']['num_labels'] = 19293
for mode in ('latest', 'best', 'explicit.pt'):
    scope = dict(ckpt_config_preview=preview, ckpt_state_dict_keys=['patch_embed.weight'],
        config=cfg, validate_config_compatibility=validate_config_compatibility,
        resume_opt=mode, logger=logging.getLogger('audit'), ckpt_path=Path('synthetic.pt'))
    try:
        exec(compile(ast.Module(body=[guard], type_ignores=[]), 'train_direct.py', 'exec'), scope)
        print(mode, scope['ckpt_path'])  # latest/best: None
    except ValueError:
        print(mode, 'ValueError')       # explicit.pt
```

### Worker allocation (issue 9)

This reads the existing cache through mmap and writes only a temporary 10,000-row IPC. It does not instantiate the full production loaders.

```python
import pickle, tempfile
from pathlib import Path
import numpy as np
import pyarrow as pa
from dataset_loader import ArrowMetadataAccessor

cache = Path('logs/metadata_cache/2df7eb3c2bc0d784.metadata.arrow')
with pa.memory_map(str(cache), 'r') as source:
    full = pa.ipc.open_file(source).read_all()
    sample = full.slice(0, 10000)
    with tempfile.TemporaryDirectory(prefix='oo_arrow_audit_') as td:
        path = Path(td)/'sample.arrow'
        with pa.OSFile(str(path), 'wb') as sink:
            with pa.ipc.new_file(sink, sample.schema) as writer:
                writer.write_table(sample)
        indices = np.arange(9500, dtype=np.uint32)
        parent = ArrowMetadataAccessor(sample.take(indices), path, indices)
        worker = pickle.loads(pickle.dumps(parent))
        before = pa.total_allocated_bytes()
        worker[0]
        print('allocated', pa.total_allocated_bytes()-before)
        print('table bytes', worker._table.nbytes)
        print('same row', parent[123] == worker[123])
        print('six-worker estimate GiB', 6*.95*full.nbytes/2**30)
```

## Working-tree fingerprints

SHA-256 of the files used for the findings, recorded before writing this report. These distinguish the audited uncommitted state from HEAD. Line numbers refer to this state.

```text
8a2dd02edc2a884513350c5a45decf9e7a098f3b1614c741ecd560f1bc832e6b  train_direct.py
e2f9a503885ee1170d17599ae368d9367a2e1fc7e1e222182f745d3ef548d10b  training_utils.py
f54be9a9554742805aef9c45330fe0184da80a43f6ab9e9e32bdcb2a3b4f161a  dataset_loader.py
0046187ecc8470e96f8364b56c838ff8731a71bbcca7d0b7e77d024c63ee4d90  loss_functions.py
9fd31164c12b51eb4fe83897da5c8b5393aa1ac9ef3d5ae193704ac8b1667fd1  model_architecture.py
87154db3feecd2b3c66eccd598ab8b6d1f2bf733195760f1395935c19053b3e7  evaluation_metrics.py
4de7999b7529a17a723536b5dd1c4140324b859127c81380fda3a9b0041470f2  asl_telemetry.py
3f8ff068dc020360d7e8baa25c6be71f718b56b4827078f8d8e4cd3b11547cbf  stop_conditions.py
74b777da4d75fa55dbcc46789eb71065d05100d2eefe62964ff188b81c669446  Configuration_System.py
5cbba17ee6fc9d00ccbd5f2bfb52bbf11cfbabb06bb39780dfae625e35f95d96  configs/unified_config.yaml
3030fcaef2bf23fe107b0ee1298808dc8ae8c468ac7ef32b61aa66706fbb404e  todos/v2-plan.md
```
