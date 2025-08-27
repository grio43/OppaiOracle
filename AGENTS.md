# Agent Instructions

This document provides guidance for AI agents working in this repository.

## Project Overview

OppaiOracle, also known as **MAID (Model for AI-based Detection)**, is a PyTorch-based system for training, evaluating, and deploying image-tagging models. It supports configuration-driven experimentation, ONNX export, and a suite of utilities for data handling and evaluation.

## Key Files and Directories

- `AGENTS.md`: This document.
- `README.md` and `maid/README.md`: High-level project documentation.
- `configs/`: Configuration files, including the primary `unified_config.yaml`.
- `train_direct.py`: Main training entry point.
- `Inference_Engine.py`: PyTorch inference logic.
- `onnx_infer.py`: Inference using exported ONNX models.
- `ONNX_Export.py`: Exports trained models to ONNX format.
- `Configuration_System.py`: Validates and manages configuration files.
- `validation_loop.py`: Core validation logic used during training.
- `HDF5_loader.py`: Utilities for loading datasets stored in HDF5 format.
- `TEst and review/`: Scripts for model evaluation and result visualization.
- `utils/`, `scripts/`, `tools/`: Shared utilities, helper scripts, and calibration tools.

## Development Workflow

### 1. Environment Setup

Install project dependencies:

```bash
pip install -r requirements.txt
```

### 2. Configuration

Validate your configuration before running other scripts. This ensures all settings in `configs/unified_config.yaml` are correct.

```bash
python Configuration_System.py validate configs/unified_config.yaml
```

### 3. Training

Train the model using the main training script:

```bash
python train_direct.py --config configs/unified_config.yaml
```

### 4. Inference

After training, you can run inference using the PyTorch model or an exported ONNX model.

- **PyTorch Inference:**
  ```bash
  python Inference_Engine.py --config configs/unified_config.yaml --image <path_to_image>
  ```
- **ONNX Export:**
  ```bash
  python ONNX_Export.py --config configs/unified_config.yaml
  ```
- **ONNX Inference:**
  ```bash
  python onnx_infer.py --config configs/unified_config.yaml --image <path_to_image>
  ```

## Evaluation Workflow

This project has a comprehensive suite of tools for evaluation. The typical workflow is as follows:

### 1. Run Batch Evaluation

Use `batch_evaluate.py` to evaluate a trained model against a directory of images and their corresponding ground-truth tags. This script produces a JSONL file with detailed results for each image.

```bash
# Example command (adjust paths as needed)
python TEst\ and\ review/batch_evaluate.py \
    --model-path /path/to/your/model.pt \
    --image-dir /path/to/images \
    --json-dir /path/to/json_tags \
    --output /path/to/results.jsonl
```

### 2. Monitor in Real-Time (Optional)

While the batch evaluation is running, you can use `live_viewer.py` to monitor the progress in real-time. It provides a terminal-based dashboard with live metrics.

```bash
python TEst\ and\ review/live_viewer.py /path/to/results.jsonl
```

For a web-based interface, you can run:
```bash
python TEst\ and\ review/live_viewer.py /path/to/results.jsonl --mode web
```

### 3. Visualize Final Results

Once the evaluation is complete, use `visualize_results.py` to generate plots and a summary report from the output JSONL file. This helps in analyzing model performance.

```bash
python TEst\ and\ review/visualize_results.py \
    --results /path/to/results.jsonl \
    --outdir /path/to/visualization_output
```

## Testing

This project does not have a formal test suite. Before committing any changes, ensure all Python files are syntactically correct by running a compilation check. This command is designed to handle filenames with spaces.

```bash
git ls-files '*.py' | xargs -I {} python -m py_compile "{}"
```

When modifying a script, run it with sample data or its `--help` flag to verify its behavior and prevent runtime errors.
