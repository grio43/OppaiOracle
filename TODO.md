TODO — Orphans, Critical and Major Fixes

Scope: Only orphan code, critical errors, and major errors that block or meaningfully degrade training, validation, export, or inference.

## Critical Errors

### ONNX export missing import: `torch.nn.functional as F`

- **File**: `ONNX_Export.py:1`
- **Symptom**: `NameError: F` in `InferenceWrapper.forward()` when calling `F.pad`/`F.interpolate`.
- **Fix**: Add `import torch.nn.functional as F` near the other imports so the functional API is available. Without this import, calls to `F.pad`/`F.interpolate` will fail during export.
- **2025 context**: The PyTorch 2.8 ONNX exporter still relies on the functional API for in‑graph pre‑/post‑processing. Exporting ops like pad, resize and normalise is supported from opset 18 onward, but they still require the `F` alias. When resizing, ensure `align_corners=False` to silence deprecation warnings and produce deterministic results:contentReference[oaicite:0]{index=0}.
- **Code example**:

  ```python
  # In ONNX_Export.py
  import torch.nn.functional as F  # required for pad/interpolate functions

  def forward(self, x: torch.Tensor) -> torch.Tensor:
      # pad to a square and resize for ONNX export
      x = F.pad(x, (0, 0, pad_h, pad_w))
      x = F.interpolate(x, size=(self.cfg.img_size, self.cfg.img_size), mode="bilinear", align_corners=False)
      return x
  ```
- **References**: ONNX opset 18–19 support for image operations and PyTorch ONNX exporter notes.

### ONNX inference alpha compositing uses incorrect mask

- **File**: `onnx_infer.py:102`
- **Symptom**: `PIL.Image.paste` is called with `mask=img` (an RGBA image) instead of using its alpha channel, causing incorrect transparency handling and errors on strict builds.
- **Fix**: Use the alpha channel of the RGBA image when pasting: `background.paste(img, mask=img.split()[-1])`. The mask must be a single‑channel (L/1) image:contentReference[oaicite:1]{index=1}.
- **2025 context**: Pillow 10/11 retain the same API; the `mask` argument to `Image.paste` must be a single‑band image. Passing the full RGBA image is invalid and may raise a `ValueError`:contentReference[oaicite:2]{index=2}.
- **References**: Pillow `Image.paste` documentation:contentReference[oaicite:3]{index=3}.

### Fragile type annotation when monitoring is unavailable

- **File**: `Inference_Engine.py:107`
- **Symptom**: `InferenceConfig` annotates `monitor_config: Optional[MonitorConfig]`, but the `MonitorConfig` import is optional. On Python 3.12, evaluating the annotation when the import fails raises a `NameError` at class definition time.
- **Fix**: Use string forward references and conditional imports:

  ```python
  from typing import Optional, TYPE_CHECKING
  if TYPE_CHECKING:
      from .monitoring import MonitorConfig

  class InferenceConfig(BaseModel):
      monitor_config: Optional["MonitorConfig"] = None
  ```

  Alternatively, add `from __future__ import annotations` at the top of the file to postpone evaluation of annotations.
- **2025 context**: Postponed evaluation of annotations is still opt‑in in Python 3.12; using strings or enabling the future import prevents runtime name errors. Forward references are required whenever the target may not be imported:contentReference[oaicite:4]{index=4}.
- **References**: Python typing module documentation on `TYPE_CHECKING` and postponed annotation evaluation:contentReference[oaicite:5]{index=5}.

### Hard‑coded absolute path fallback for vocabulary

- **File**: `ONNX_Export.py:155`
- **Symptom**: The code attempts to load `/media/andrewk/qnap-public/workspace/OppaiOracle/vocabulary.json` when normal paths fail, making the export non‑portable.
- **Fix**: Remove the hard‑coded path and surface a clear error instructing the user to provide a `vocabulary.json` via `export.vocab_dir` or embed it in the checkpoint/ONNX metadata. When exporting, load the vocabulary from configuration or embed it directly into the ONNX model’s metadata using `model.metadata_props.add()`.
- **2025 context**: The ONNX format supports arbitrary key‑value metadata. Best practice is to embed class names or vocabulary in the model so that it travels with the checkpoint. For example, you can attach a dictionary of class names using the Python API:

  ```python
  import json
  import onnx

  model = onnx.load("model.onnx")
  class_names = {0: "tag0", 1: "tag1", 2: "tag2"}
  meta = model.metadata_props.add()
  meta.key = "class_names"
  meta.value = json.dumps(class_names)
  onnx.save(model, "model.onnx")
  ```

  This avoids searching for external JSON files and makes the export self‑contained:contentReference[oaicite:6]{index=6}.
- **References**: ONNX custom metadata examples:contentReference[oaicite:7]{index=7}.

### Packaging constraints need platform‑specific install guidance

- **File**: `requirements.txt:29–30`
- **Symptom**: The project pins `torch>=2.8.0` and `torchvision>=0.23.0`. Installing these packages blindly can fail because wheels differ by Python version, OS and CUDA. Users might hit installation errors or get CPU‑only builds unintentionally.
- **Fix**: Document explicit installation commands for common environments and pin a tested pair. For example:

  * **CUDA 12.x on Linux (PyTorch 2.8.0 + torchvision 0.23.0)**

    ```bash
    pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu121
    ```

  * **CUDA 11.x** — use the cu118 or cu121 index corresponding to your driver; see the PyTorch selector for the full matrix:contentReference[oaicite:8]{index:8}.

  * **CPU‑only** (no NVIDIA GPU):

    ```bash
    pip install torch==2.8.0 torchvision==0.23.0
    ```

  Always ensure your Python version (≥3.9) matches the supported wheels, and install `torchmetrics` separately when using Lightning metrics.
- **2025 context**: PyTorch 2.8 requires Python 3.9+ and provides wheels for CPU and CUDA 12.x/11.x. Use the official installer matrix to select the correct index URL:contentReference[oaicite:9]{index=9}:contentReference[oaicite:10]{index=10}. For GPU inference with ONNX Runtime, match your CUDA major version (11.x or 12.x) to avoid mismatched cuDNN versions:contentReference[oaicite:11]{index=11}.
- **References**: PyTorch “Get Started” installer matrix:contentReference[oaicite:12]{index=12}:contentReference[oaicite:13]{index=13} and ONNX Runtime CUDA compatibility table:contentReference[oaicite:14]{index=14}.

### Unused helper with incorrect signature

- **File**: `onnx_infer.py:115`
- **Symptom**: `_preprocess_simple(image_path, image_size, mean, std)` calls an undefined `_preprocess(...)` and is never invoked.
- **Fix**: Delete `_preprocess_simple` or implement it properly. Since the ONNX graph already performs preprocessing, maintaining a second code path invites divergence; prefer a single preprocessing implementation.
- **2025 context**: Consolidating preprocessing reduces maintenance and ensures parity between training, export and inference.

## Major Issues

### Lightning training path is incomplete/broken (treat as orphan unless finished)

- **Files**: `train_lightning.py`, `lightning_module.py:28,43,62`
- **Problems**:
  - The `LightningDataModule` returns empty dataloaders, so no training data is ever consumed.
  - The `LightningModule` expects outputs `outputs['tag']` and `outputs['rating']`, but the model returns `{'tag_logits', 'rating_logits'}`, causing `KeyError`.
- **Options**:
  - Mark the Lightning training path as experimental and exclude it from documentation until it is complete.
  - Or, fix the implementation: create a real `LightningDataModule` that wraps the existing `dataset_loader.create_dataloaders()` and yield batches as `(images, targets_dict)`, and modify the `LightningModule` to use the correct keys.
- **2025 fix guidance**:
  - **Implement a proper DataModule** – A `LightningDataModule` should implement `setup()`, `train_dataloader()`, `val_dataloader()` and optionally `test_dataloader()`. Use the existing dataset loader to populate training and validation datasets and wrap them in `DataLoader` instances:contentReference[oaicite:15]{index=15}. For example:

    ```python
    class OppaiDataModule(L.LightningDataModule):
        def __init__(self, cfg):
            super().__init__()
            self.cfg = cfg

        def setup(self, stage: str = None):
            loaders = dataset_loader.create_dataloaders(self.cfg)
            self.train_loader = loaders['train']
            self.val_loader = loaders['val']

        def train_dataloader(self):
            return self.train_loader

        def val_dataloader(self):
            return self.val_loader
    ```

  - **Align outputs** – In the `training_step`, rename the keys returned by the model to match those expected by the loss functions and metrics:

    ```python
    def training_step(self, batch, batch_idx):
        images, targets = batch
        outputs = self.model(images)
        tag_logits = outputs['tag_logits']
        rating_logits = outputs['rating_logits']
        # compute losses using targets['tag_labels'], targets['rating_labels']
        ...
    ```

  - **Use TorchMetrics for multi‑label classification** – TorchMetrics provides `MultilabelF1Score` and other metrics. Instantiate them with `num_labels` equal to your vocabulary size and set a probability threshold:

    ```python
    from torchmetrics.classification import MultilabelF1Score
    self.train_f1 = MultilabelF1Score(num_labels=len(self.vocab), average='macro', threshold=0.5)
    self.valid_f1 = MultilabelF1Score(num_labels=len(self.vocab), average='macro', threshold=0.5)
    ```

    During each step, update the metric with `(logits, targets)` and log it at the end of the epoch.
- **References**: Lightning DataModule documentation:contentReference[oaicite:17]{index=17} and a multi‑label classification example demonstrating `MultilabelF1Score`.

### Epoch‑based warmup misconfigured

- **File**: `configs/unified_config.yaml: training.warmup_steps`
- **Context**: The scheduler used (`CosineAnnealingWarmupRestarts`) advances once per **epoch** (not per batch). Setting `warmup_steps` to `10000` therefore applies a 10 k‑epoch warmup, keeping the learning rate near zero for the entire run.
- **Fix**: Set `training.warmup_steps` to a small integer (3–10) so the learning rate ramps up over a few epochs. For example:

  ```yaml
  training:
    warmup_steps: 5  # number of epochs to warm up
  ```

- **2025 context**: Warm‑restart schedulers like cosine annealing with warm restarts use parameters `T_0` (number of epochs until first restart) and `T_mult` to determine cycle lengths. Shorter cycles are suitable for small datasets, while very large values can stall training:contentReference[oaicite:19]{index=19}. Always configure the scheduler to step per epoch or per batch consistently:contentReference[oaicite:20]{index=20}.
- **References**: Milvus article on implementing cosine annealing with warm restarts:contentReference[oaicite:21]{index=21} and PyTorch scheduler documentation:contentReference[oaicite:22]{index=22}.

### Non‑portable default data path enabled

- **File**: `configs/unified_config.yaml: data.storage_locations[0].path`
- **Symptom**: The first storage location points to a machine‑specific absolute path and is marked `enabled: true`, causing “No data” errors on other systems.
- **Fix**: Disable this entry by default or replace it with a placeholder path (e.g., `path: /path/to/your/dataset` and `enabled: false`). Document how to set exactly one `enabled: true` path via environment variables or CLI arguments.
- **2025 context**: Users often run training in Docker containers or cloud environments. Using relative paths or environment variables improves portability and prevents accidental leaks of local directory names.

### Duplicated path utilities increase confusion

- **Files**: `utils/metadata_ingestion.py:47` and `utils/path_utils.py:26`
- **Symptom**: Two implementations of `validate_image_path`/`safe_join` exist. Only the one in `utils/path_utils` is used by the data loaders.
- **Fix**: Remove the duplicate implementations from `metadata_ingestion.py` and import the functions from `utils/path_utils` wherever they are needed. Keep path sanitisation logic in one place to avoid inconsistencies.
- **2025 context**: Centralised path utilities reduce bugs and make it easier to audit path traversal protections.

### Logging couples to dataset module

- **File**: `utils/logging_setup.py:15`
- **Symptom**: Imports `dataset_loader.CompressingRotatingFileHandler` solely to get a log handler, thereby pulling in heavy dataset dependencies during logger initialisation.
- **Fix**: Move `CompressingRotatingFileHandler` into a lightweight module under `utils/` (e.g., `utils/logging_handlers.py`). In `logging_setup.py`, import it from this new module. This decouples logging from the dataset module and prevents side effects when configuring logging.
- **2025 context**: Avoid side‑effect imports in generic code paths. Keeping logging independent makes it easier to use the utilities in non‑training scripts.

## Orphan/Dead Code

- `_preprocess_simple` in `onnx_infer.py` is unused and has the wrong signature. Delete it or make it call the correct `_preprocess` function with matching arguments.
- `vocab_utils.py` is only referenced by `train_lightning.py`. If the Lightning path is deferred, remove this file or migrate its functions into the main training utilities.
- Placeholder classes/functions in `dataset_loader.py` (`AugmentationStats`, `validate_dataset`) are unused. Implement them or remove them from the public API. Keeping unused placeholders confuses new contributors and can trigger import‑sanity failures.

## Preflight/Config Gating (non‑code but critical to avoid runtime failures)

- **Vocabulary**: Provide a real `vocabulary.json` (no `tag_###` placeholders) before starting training or export. You can generate this file with `vocabulary.create_vocabulary_from_datasets(...)` or embed the vocabulary directly in the checkpoint/ONNX metadata using the method shown above. Failing to provide a proper vocabulary will cause immediate errors when loading the model.
- **Orientation flips**: When `data.random_flip_prob > 0`, ensure that `configs/orientation_map.json` exists and maps each tag to its flipped equivalent. If no map is provided, set `data.strict_orientation_validation=false` to disable orientation checks. Default to conservative safety settings to avoid training on incorrectly flipped labels.

## Notes

- The “validation header glitch” mentioned in the runbook is not present; `validation_loop.py` begins with a proper shebang and docstring.

## Quick References (2025)

- **Pillow paste mask** – `Image.paste` requires a single‑band mask (alpha channel):contentReference[oaicite:23]{index=23}.
- **Python typing forward refs/`TYPE_CHECKING`** – Use string annotations or `from __future__ import annotations` to avoid evaluation errors:contentReference[oaicite:24]{index=24}.
- **PyTorch install matrix (2.8)** – Use the official selector to choose the correct wheel and index URL:contentReference[oaicite:25]{index=25}:contentReference[oaicite:26]{index=26}.
- **ONNX Runtime GPU providers** – Match ONNX Runtime CUDA version (11.x/12.x) with your PyTorch CUDA version to avoid cuDNN mismatches:contentReference[oaicite:27]{index=27}.
- **Lightning DataModule** – A `LightningDataModule` encapsulates data preparation and dataloaders:contentReference[oaicite:28]{index=28}.
- **TorchMetrics multi‑label F1** – Use `MultilabelF1Score(num_labels, average="macro", threshold=…)` to compute F1 for multi‑label tasks.
- **Cosine annealing warm restarts** – Tune `T_0` and `T_mult` values and keep warmup epochs small to prevent a long warm‑up phase:contentReference[oaicite:30]{index=30}:contentReference[oaicite:31]{index=31}.

