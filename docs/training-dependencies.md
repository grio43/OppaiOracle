# Training dependencies

The Windows/NVIDIA training stack is pinned as a unit in `requirements.txt`:

| Package | Version |
| --- | --- |
| PyTorch | 2.14.0 (CUDA 13.0 wheel) |
| torchvision | 0.29.0 (CUDA 13.0 wheel) |
| triton-windows | 3.8.0.post28 |
| bitsandbytes | 0.50.2 |

Update PyTorch, torchvision and Triton together. PyTorch 2.14 requires the
Triton 3.8 family; independently upgrading Triton can break Inductor compilation.
The Windows post28 release includes fixes for misaligned accesses on SM120/121,
which is relevant to the RTX 5090. These fixes do not establish a training
throughput gain for this model.

Sources: [PyTorch 2.14 release](https://github.com/pytorch/pytorch/releases/tag/v2.14.0),
[Triton Windows post28 release](https://github.com/triton-lang/triton-windows/releases/tag/v3.8.0-windows.post28),
[bitsandbytes 0.50.2 release](https://github.com/bitsandbytes-foundation/bitsandbytes/releases/tag/0.50.2).

The optimizer factory omits the removed `block_wise=True` constructor argument:
bitsandbytes 0.50 now always uses block-wise 8-bit states. The tag-head and
position-embedding overrides still keep their optimizer state in fp32.
Block-mask creation reuses `torch.compile(create_block_mask)` instead of the
deprecated private `_compile=True` argument, retaining the same mask semantics.
Its host dispatch is a graph boundary (`torch.compiler.disable(recursive=False)`)
so the mask builder can compile separately: inlining it into the dynamic image
graph reproduced an Inductor `CantSplit` error on PyTorch 2.14. The inner mask
kernels remain compiled. This uses the existing `compile_fullgraph: false`
training setting.

## Install or update

Close processes using `L:\Dab\payton_env` before replacing its packages. Save the
current package versions if updating an existing environment:

```powershell
L:\Dab\payton_env\Scripts\python.exe -m pip freeze --all > previous-dependencies.txt
.\payton_env.ps1 -InstallDeps
.\Start_AI_Training.ps1 -CheckTrainingStack
```

`-InstallDeps` installs `requirements-training-cu130.txt` from the official
PyTorch CUDA 13.0 index first, then the remaining `requirements.txt` packages,
and finally runs `pip check`. Native command failures stop the installer.
Keep the torch/torchvision pins in both requirements files aligned.
Other platforms can install `requirements.txt` with their appropriate PyTorch
wheel index; the Windows Triton distribution has a platform marker, and Linux
PyTorch manages its upstream Triton dependency.

The old environment included **torchaudio 2.11**, which requires torch 2.11.
OppaiOracle does not use torchaudio, and the CUDA index has no matching 2.14
release. If it is still installed in a legacy environment, remove it before
upgrading:

```powershell
L:\Dab\payton_env\Scripts\python.exe -m pip uninstall torchaudio
```

New environments default to Python 3.12, matching `pyproject.toml`. Activation
preserves an existing interpreter (including the tested legacy Python 3.11.9
environment); dependency updates do not replace Python in place. An explicit
`-PythonVersion` is checked against the selected environment. `-PythonExe` can
select the interpreter when creating a new environment.

## Verify the kernels

`Start_AI_Training.ps1 -CheckTrainingStack` uses the launcher's MSVC and Windows
SDK setup and runs `tools/check_training_stack.py`. It reads the YAML model
geometry and compile settings, then uses two synthetic images, two transformer
blocks and 1,024 labels. It does not read production images, prepare vocabulary,
or create training checkpoints.

The probe checks compiled Flex Attention logits, ASL loss and representative
backward gradients against the model's SDPA path under bfloat16 autocast. It
exercises gradient checkpointing, masked labels, AdamW8bit with fp32 tag-head
state, and validation at batch sizes two and one. Dropout is disabled for the
numerical comparison. Compiler exceptions and nonfinite gradients fail the probe.

```powershell
.\Start_AI_Training.ps1 -CheckTrainingStack -TrainingArgs @('--output', 'logs/training-stack.json')
L:\Dab\payton_env\Scripts\python.exe -B -m unittest test_v2_pipeline test_training_audit test_training_readiness test_rating_system
```

The first step includes compilation/autotuning. The reported median uses the
remaining steps. This is a compatibility probe, not a full training benchmark:
the reduced depth, synthetic labels and competing GPU work affect timing.
Measure the complete model with the final vocabulary on an otherwise idle GPU
before choosing the production batch size or claiming an end-to-end speedup.

No extra attention implementation, precision change, or training-recipe change
is needed for this dependency update. The active compile mode remains
`max-autotune-no-cudagraphs`.

## Local validation

Verified on 2026-09-07 with the RTX 5090, driver 610.62, and the existing Python
3.11.9 environment:

- `pip check` passed; a dry-run of `requirements.txt` required no further installs.
- All 35 V2/training-audit/readiness/rating regression tests passed.
- Two focused model/optimizer tests passed with deprecations treated as errors.
- The final compiled probe passed training, SDPA comparisons, fp32/8-bit optimizer
  state checks, and full/partial-batch validation. Relative gradient differences
  for QKV, MLP and the tag head were 0.74%, 0.51% and 0.45%.
- Python 3.12 Windows wheels for all four pinned packages resolved successfully;
  the actual GPU execution above used Python 3.11.9.

**A performance improvement has not been established.** The two-layer synthetic
probe measured a median warm step of 12.81 ms on torch 2.11 / Triton 3.6 /
bitsandbytes 0.49, and 73.58 ms on the final updated stack. Other GPU work was
active, and utilization reached 92–93% during the later probes, so this was not
an isolated A/B benchmark and cannot attribute the difference to the dependency
update. It also does not measure the complete production model. An idle-GPU
comparison is still needed before making a throughput claim.

The initial upgraded compilation/autotuning step took 380 seconds; the final
probe reused kernels and took 25.83 seconds for its first step. The baseline
first step took 154 seconds. These are observed startup times with different
cache states, not controlled compiler benchmarks.

Local, untracked evidence is in `.code-review/dependency-before.json`,
`.code-review/dependency-after.json`, `.code-review/dependency-final.json`,
`.code-review/dependency-regressions.log`, and
`.code-review/dependency-mask-regressions.log`. The corresponding before/after
`pip freeze --all` snapshots are retained in `.code-review/`.
