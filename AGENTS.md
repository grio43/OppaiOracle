# Agent Instructions

This document provides guidance for AI agents working in this repository.

## Project Overview

OppaiOracle, also known as **MAID (Model for AI-based Detection)**, is a PyTorch-based system for training, evaluating, and deploying image‑tagging models. It supports configuration‑driven experimentation, ONNX export, and a suite of utilities for data handling and evaluation.

## Key Files and Directories

- `AGENTS.md`: This document.
- `README.md`: High‑level project documentation.
- `configs/`: Configuration files, including the primary `unified_config.yaml`.
- `train_direct.py`: Main training entry point.
- `validation_loop.py`: Core validation logic used during training.
- `evaluation_metrics.py`: Implements metrics consumed by the validation loop.
- `training_utils.py`: Shared helpers for the training pipeline.
- `loss_functions.py`: Custom loss definitions.
- `model_architecture.py`: Defines the neural network.
- `adan_optimizer.py` and `custom_drop_path.py`: Optional optimization and regularization components.
- `orientation_handler.py`: Handles image orientation metadata.
- `dataset_loader.py`: Utilities for loading datasets.
- `Dataset_Analysis.py`: Tools for inspecting datasets.
- `Inference_Engine.py`: PyTorch inference logic.
- `ONNX_Export.py`: Exports trained models to ONNX format.
- `onnx_infer.py`: Inference using exported ONNX models.
- `model_metadata.py`: Stores metadata about trained models.
- `sensitive_config.py`: Centralizes sensitive configuration data.
- `TEst and review/`: Scripts for model evaluation and result visualization.
- `utils/`, `scripts/`, `tools/`: Shared utilities, helper scripts, and calibration tools.
- `logs/`: Outputs from training and other runs.

## Development Workflow

### 1. Environment Setup

Install project dependencies:

```bash
pip install -r requirements.txt
```

### 2. Configuration

Validate your configuration before running other scripts to ensure all settings in `configs/unified_config.yaml` are correct:

```bash
python Configuration_System.py validate configs/unified_config.yaml
```

### 3. (Optional) Dataset Analysis

Use the dataset analysis tool to inspect your data:

```bash
python Dataset_Analysis.py --help
```

### 4. Training

Train the model using the main training script:

```bash
python train_direct.py --config configs/unified_config.yaml
```

### 5. Inference

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

### 6. Evaluation

Evaluate model performance with the validation loop and associated metrics, or use the scripts in `TEst and review/` for batch evaluation and visualization.

## Testing

This project does not have a formal test suite. Before committing any changes, ensure all Python files are syntactically correct by running a compilation check (handles filenames with spaces):

```bash
git ls-files '*.py' | xargs -I {} python -m py_compile "{}"
```

When modifying a script, run it with sample data or its `--help` flag to verify its behavior and prevent runtime errors.
