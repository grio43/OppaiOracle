# OppaiOracle (MAID) – User Manual

This manual explains how to install, configure, train, export, and run inference with OppaiOracle (aka MAID), and how to use the included tools. It also highlights key design choices and caveats observed during a full code review of the repository.

---

## 1) Quick Start

- Create a Python 3.12 environment and install deps:
  ```bash
  python3.12 -m venv .venv
  source .venv/bin/activate
  pip install -r requirements.txt
  ```
- Validate your config:
  ```bash
  python Configuration_System.py validate configs/unified_config.yaml
  ```
- Train (direct training path):
  ```bash
  python train_direct.py --config configs/unified_config.yaml
  ```
- Export to ONNX (with embedded vocabulary + metadata):
  ```bash
  python ONNX_Export.py \
    /path/to/checkpoints/best_model.pt \
    ./vocabulary.json \
    --output ./exported/model.onnx --variants full quantized --benchmark
  ```
- Inference (PyTorch):
  ```bash
  python Inference_Engine.py \
    --model ./checkpoints/best_model.pt \
    --config ./checkpoints/model_config.json \
    --vocab ./vocabulary.json
  ```
- Inference (ONNX, reads embedded vocabulary when present):
  ```bash
  python onnx_infer.py exported/model.onnx /path/to/img1.png /path/to/img2.jpg \
    --top_k 10 --threshold 0.5 --output results.json
  ```

---

## 2) Repository Structure (What Lives Where)

- Training core: `train_direct.py`, `training_utils.py`, `model_architecture.py`, `loss_functions.py`, `evaluation_metrics.py`, `dataset_loader.py`, `schedulers.py`, `safe_checkpoint.py`
- Inference: `Inference_Engine.py` (PyTorch), `onnx_infer.py` (ONNX Runtime)
- Export: `ONNX_Export.py`, `model_metadata.py`, `scripts/convert_vocab_to_metadata.py`
- Config System: `Configuration_System.py`, `configs/unified_config.yaml`, `configs/orientation_map.json`
- Vocabulary + Data Prep: `vocabulary.py`, `tag_vocabulary.py`, `utils/metadata_ingestion.py`, `utils/path_utils.py`
- Orientation: `orientation_handler.py`, `configs/orientation_map.README.md`
- Tools & Eval: `tools/calibrate_thresholds.py`, `TEst and review/visualize_results.py`, `TEst and review/live_viewer.py`, `TEst and review/batch_evaluate.py`
- Logging: `utils/logging_setup.py`

Key entry points discovered via code review:
- Training: `train_direct.py:854`, `train_lightning.py:98`
- Inference: `Inference_Engine.py:969`, `onnx_infer.py:244`
- Export: `ONNX_Export.py:1255`
- Validation: `validation_loop.py:1258`
- Dataset analysis: `Dataset_Analysis.py:1311`
- Config validate/generate: `Configuration_System.py:1826`

---

## 3) Configuration

The single source of truth is `configs/unified_config.yaml`. It covers model, data, training, inference, export, validation, and monitoring settings.

- Validate a config file:
  ```bash
  python Configuration_System.py validate configs/unified_config.yaml
  ```
- Generate example configs (optional, for reference):
  ```bash
  python Configuration_System.py generate ./config_examples
  ```
- Environment variable overrides (highest priority):
  - Format: `ANIME_TAGGER_<SECTION>__<FIELD>[__SUBFIELD...]`
  - Examples:
    ```bash
    ANIME_TAGGER_TRAINING__LEARNING_RATE=5e-5 \
    ANIME_TAGGER_DATA__BATCH_SIZE=4 \
    ANIME_TAGGER_EXPORT__QUANTIZE=true \
    python train_direct.py --config configs/unified_config.yaml
    ```
- CLI dot-notation overrides exist in the parser but note:
  - Current `train_direct.py` loads only `--config` and `--validate-only`. CLI field overrides are not applied in that script; use env vars or edit YAML.

Common invariants and tips (from `configs/README.md` and code):
- Ensure `data.image_size` is divisible by `model.patch_size`.
- Keep `data.normalize_mean/std` and `data.pad_color` consistent across training/inference/export.
- Effective batch size ≈ `data.batch_size * training.gradient_accumulation_steps * training.world_size`.

---

## 4) Data & Vocabulary

Two dataset modes are supported during training/validation (see `dataset_loader.py`):

- Manifest mode: directory with `images/`, `train.json`, `val.json`
- Sidecar mode: recursively scan per‑image `*.json` files next to images

Sidecar JSON format (see `dataset_loader.py:418`):
```json
{
  "filename": "12345.jpg",        // optional, used to derive image_id
  "tags": "red_hair smile ...",  // string or list of strings
  "rating": "general"            // optional: general/sensitive/questionable/explicit/unknown
}
```

Vocabulary management (`vocabulary.py`):
- Build from dataset annotations (keeps top‑K by frequency, ignores tags listed in `Tags_ignore.txt` if present):
  ```bash
  python vocabulary.py /path/to/annotations_root
  # writes ./vocabulary.json
  ```
- Safety: the system fails fast if the vocab contains placeholder tags like `tag_1234`. Regenerate your vocabulary if you see that error.
- During training, tag index 0 is reserved for `<PAD>` and ignored by the loss; unknown tags map to `<UNK>`.

Optional tag‑phase prep for Danbooru‑style metadata (`tag_vocabulary.py:681`):
```bash
python tag_vocabulary.py \
  --metadata_dir /path/to/jsons \
  --vocab_path ./vocabulary.json \
  --output_dir ./outputs/phased \
  --phase1_size 4000000 --total_size 8500000
```

---

## 5) Training

Direct training entry point (`train_direct.py:854`):
```bash
python train_direct.py --config configs/unified_config.yaml
```

- Reproducible run with deterministic flags:
  ```bash
  scripts/run_train_deterministic.sh --config configs/unified_config.yaml
  ```
- Key knobs to review in `unified_config.yaml`:
  - `data`: `batch_size`, `num_workers`, `image_size`, normalization, caching, orientation mapping
  - `training`: `num_epochs`, `learning_rate`, optimizer/scheduler, AMP (`amp_dtype: bfloat16`), gradient clipping, checkpoint cadence
  - `model`: ViT depth/width, `drop_path_rate`, `token_ignore_threshold`, `image_size`, `patch_size`
- Internals:
  - Model: Vision Transformer with mixed‑precision safety (`model_architecture.py`)
  - Loss: Asymmetric focal loss + rating head (`loss_functions.py`)
  - Metrics: macro/micro F1 and mAP (`evaluation_metrics.py`)
  - Dataloader: manifest or sidecar JSON mode; optional LMDB L2 cache (`dataset_loader.py`)

Checkpoints & logs:
- Checkpoints and metrics are saved under `logs/` by default. See `CheckpointManager` in `training_utils.py` and JSON metric snapshots in `logs/checkpoints/`.
- TensorBoard and optional Weights & Biases logging are wired via `monitor` settings in `unified_config.yaml` and `utils/logging_setup.py`.

---

## 6) Validation & Evaluation

Validation runner (`validation_loop.py:1258`):
```bash
python validation_loop.py \
  --checkpoint ./logs/checkpoints/best.pt \
  --data-dir /path/to/images \
  --json-dir /path/to/jsons \
  --vocab-path ./vocabulary.json \
  --mode full --batch-size 64 --output-dir ./validation_results
```
- Modes: `full`, `fast`, `tags`, `hierarchical` (see parser at `validation_loop.py:1181`).
- Outputs include metrics summary, optional per‑image predictions and plots.

Batch evaluation and visualization (templates in `TEst and review/`):
- Live viewer (terminal or simple web):
  ```bash
  python "TEst and review/live_viewer.py" /path/to/results.jsonl --mode terminal
  ```
- Plot/HTML report from a summary JSON(L):
  ```bash
  python "TEst and review/visualize_results.py" --results ./results/evaluation_results.json \
    --outdir ./results/plots
  ```
- Note: `TEst and review/batch_evaluate.py` currently contains hard‑coded paths; treat it as an example you copy‑edit.

---

## 7) Inference

PyTorch inference (`Inference_Engine.py:969`):
```bash
python Inference_Engine.py \
  --model ./checkpoints/best_model.pt \
  --config ./checkpoints/model_config.json \
  --vocab ./vocabulary.json
```
- Loads preprocessing defaults from `configs/inference_config.yaml` if present; otherwise uses built‑ins and tries to pull select defaults from `unified_config.yaml`.
- Optional cache and monitoring controls exist in the internal `InferenceConfig`.

ONNX Runtime inference with embedded vocab (`onnx_infer.py:244`):
```bash
python onnx_infer.py exported/model.onnx img1.png img2.jpg \
  --top_k 10 --threshold 0.5 --output results.json \
  --providers CUDAExecutionProvider CPUExecutionProvider
```
- If the model lacks embedded vocabulary metadata, pass `--vocab ./vocabulary.json`.
- Output is a standardized JSON with `metadata` and `results` (see `schemas.py`).

---

## 8) Export to ONNX

`ONNX_Export.py` reads defaults from `unified_config.yaml` and writes inference‑ready ONNX with metadata:
```bash
python ONNX_Export.py \
  ./logs/checkpoints/best_model.pt \
  ./vocabulary.json \
  --output ./exported/model.onnx \
  --variants full quantized \
  --opset 19 --optimize --quantize --benchmark
```
- Variants: `full`, `quantized`
- Embeds vocabulary and preprocessing parameters (mean/std/image_size/patch_size) into ONNX model metadata when possible.
- Optional: produce metadata JSON from a vocab file without exporting a model:
  ```bash
  python scripts/convert_vocab_to_metadata.py ./vocabulary.json --output vocab_metadata.json
  ```

---

## 9) Orientation Handling

- Safety‑aware flip support uses `configs/orientation_map.json` (see docs in `configs/orientation_map.README.md`).
- Configure via `data.orientation_*` in `unified_config.yaml`:
  - `orientation_map_path`, `orientation_safety_mode` (`conservative|balanced|permissive`), `random_flip_prob`, `skip_unmapped`, `strict_orientation_validation`.
- During sidecar loading, tags are swapped on flip using explicit/regex mappings; risky tags are vetoed per safety setting.

---

## 10) Tools

- Per‑tag threshold calibration (`tools/calibrate_thresholds.py`):
  ```bash
  python tools/calibrate_thresholds.py \
    --probs ./artifacts/val_probs.npz \
    --labels ./artifacts/val_labels.npz \
    --beta 1.0 \
    --out ./artifacts/thresholds.json
  ```
- Deterministic runner wrapper (`scripts/run_train_deterministic.sh`) to set CUDA/cuBLAS flags and seed.
- Dataset analysis (`Dataset_Analysis.py:1311`):
  ```bash
  python Dataset_Analysis.py /path/to/dataset --output-dir ./dataset_analysis --num-workers 8
  ```

---

## 11) Troubleshooting & Best Practices

- CUDA OOM: lower `data.batch_size`, increase `training.gradient_accumulation_steps`, or reduce image size.
- Placeholder tags detected (e.g., `tag_1234`): regenerate `vocabulary.json` from annotations.
- ONNX model missing embedded vocab: provide `--vocab` to `onnx_infer.py`.
- Mask errors like “masks all keys”: ensure `data.image_size` divides `model.patch_size` and padding masks are correct.
- Sidecar mode not finding images: each JSON’s directory is used to resolve images; make sure filenames/paths are consistent.
- Logging too verbose: adjust `log_level` in `unified_config.yaml` or set `runtime/monitor` knobs.

Pre‑commit sanity check (no formal tests in repo):
```bash
git ls-files '*.py' | xargs -I {} python -m py_compile "{}"
```

---

## 12) Known Gaps (from Code Review)

- `train_direct.py` parses CLI overrides but does not apply them to the loaded config; rely on YAML edits or env vars.
- `TEst and review/batch_evaluate.py` uses hard‑coded paths; adapt locally before running.
- `configs/inference_config.yaml` isn’t present by default; `Inference_Engine.py` falls back to built‑ins and unified config.
- Some scripts assume CUDA providers by default; set `--providers CPUExecutionProvider` for CPU‑only environments when using `onnx_infer.py`.

---

## 13) End‑to‑End Examples

Minimal train → export → infer (ONNX):
```bash
# 1) Validate config
python Configuration_System.py validate configs/unified_config.yaml

# 2) Ensure vocabulary exists
python vocabulary.py /data/annotations_root

# 3) Train
python train_direct.py --config configs/unified_config.yaml

# 4) Export ONNX
python ONNX_Export.py ./logs/checkpoints/best_model.pt ./vocabulary.json \
  --output ./exported/model.onnx --variants full --opset 19 --optimize

# 5) Run ONNX inference
python onnx_infer.py ./exported/model.onnx ./sample1.png ./sample2.jpg \
  --top_k 15 --threshold 0.35 --output ./results/preds.json
```

Minimal train → PyTorch inference:
```bash
python Inference_Engine.py \
  --model ./logs/checkpoints/best_model.pt \
  --config ./logs/checkpoints/model_config.json \
  --vocab ./vocabulary.json
```

---

## 14) Appendix – Where to Look in Code

- CLI parsers and options:
  - `train_direct.py:821`, `onnx_infer.py:142`, `ONNX_Export.py:1165`, `validation_loop.py:1181`, `Dataset_Analysis.py:1249`, `Inference_Engine.py:969`
- Metrics: `evaluation_metrics.py:1`
- Losses: `loss_functions.py:1`
- ViT model: `model_architecture.py:1`
- Data loading and caching: `dataset_loader.py:1`, `dataset_loader.py:728`
- Config loader (env/CLI overrides): `Configuration_System.py:1464` (env), `Configuration_System.py:1544` (args)
- Orientation docs: `configs/orientation_map.README.md:1`, mapping: `configs/orientation_map.json:1`

Happy training!

