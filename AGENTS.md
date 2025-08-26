# Agent Instructions

This document provides guidance for AI agents working in this repository.

## Project Overview

OppaiOracle, also known as **MAID (Model for AI-based Detection)**, is a PyTorch-based system for training, evaluating, and deploying image-tagging models. It supports configuration-driven experimentation, ONNX export, FastAPI serving, and a suite of utilities for data handling and evaluation.

## Key Files and Directories

- `AGENTS.md`: this document.
- `README.md` and `maid/README.md`: high-level project documentation.
- `configs/`: configuration files, including the primary `unified_config.yaml`.
- `train_direct.py`: main training entry point.
- `Inference_Engine.py`: PyTorch inference logic and FastAPI application.
- `onnx_infer.py`: inference using exported ONNX models.
- `ONNX_Export.py`: exports trained models to ONNX format.
- `Configuration_System.py`: validates and manages configuration files.
- `validation_loop.py`: core validation logic used during training.
- `HDF5_loader.py`: utilities for loading datasets stored in HDF5 format.
- `logs/`, `scripts/`, `tools/`, `TEst and review/`, `utils/`: logging, helper scripts, calibration tools, evaluation/visualization scripts, and shared utilities.

## Development Workflow

### 1. Environment Setup

Install project dependencies:

```bash
pip install -r requirements.txt
```

### 2. Configuration

Validate configuration before running other scripts:

```bash
python Configuration_System.py validate configs/unified_config.yaml
```

### 3. Training

Train the model using:

```bash
python train_direct.py --config configs/unified_config.yaml
```

Adjust arguments as needed for your experiment.

### 4. Export and Inference

- Export to ONNX:

  ```bash
  python ONNX_Export.py --config configs/unified_config.yaml
  ```
- PyTorch inference:

  ```bash
  python Inference_Engine.py --config configs/unified_config.yaml --image <path>
  ```
- ONNX inference:

  ```bash
  python onnx_infer.py --config configs/unified_config.yaml --image <path>
  ```
- Serve the FastAPI API:

  ```bash
  uvicorn Inference_Engine:app --host 0.0.0.0 --port 8000
  ```

### 5. Evaluation

`TEst and review/batch_evaluate.py` evaluates a trained model against a directory of images and associated JSON tag files, producing aggregate metrics and per-image results.

### 6. Additional Tools

- `Dataset_Analysis.py`: compute dataset statistics or visualizations.
- `scripts/`: assorted helper scripts.
- `tools/`: calibration and other utilities.

## Testing

This project has no formal test suite. For code changes, ensure Python files compile by running:

```bash
python -m py_compile $(git ls-files '*.py')
```

Run relevant scripts with sample data when possible to verify behavior.
