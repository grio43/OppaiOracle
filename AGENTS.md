# Agent Instructions

This document provides guidance for AI agents working in this repository.

## Project Overview

This repository contains the code for **MAID (Model for AI-based Detection)**, a deep learning project built with PyTorch. The primary purpose of this project is to train, evaluate, and deploy a model for AI-based detection tasks.

## Key Files and Directories

-   `AGENTS.md`: This file. Provides guidance for AI agents.
-   `Configuration_System.py`: Handles configuration management.
-   `Dataset_Analysis.py`: Scripts for analyzing datasets.
-   `Evaluation_Metrics.py`: Defines evaluation metrics for the model.
-   `HDF5_loader.py`: Loads data from HDF5 files for training and evaluation.
-   `Inference_Engine.py`: Core engine for running model inference.
-   `model_architecture.py`: Defines the neural network architecture for the model.
-   `ONNX_Export.py`: Exports the trained PyTorch model to the ONNX format for optimized inference.
-   `train_direct.py`: The main script for training the model.
-   `validation_loop.py`: Contains the logic for the model validation loop.
-   `requirements.txt`: A list of the Python dependencies for this project.
-   `configs/`: Contains configuration files, including `unified_config.yaml`, which is the main config file for training.
-   `TEst and review/`: Contains scripts for testing, evaluation, and visualization of results.
-   `scripts/`: Contains various utility scripts.
-   `tools/`: Contains tools for tasks like model calibration.
-   `utils/`: Contains utility functions used across the project.

## Development Workflow

### 1. Environment Setup

To get started, install the required dependencies from `requirements.txt`:

```bash
pip install -r requirements.txt
```

### 2. Training the Model

The main training script is `train_direct.py`. To run it, you need to provide a configuration file. The primary config file is `configs/unified_config.yaml`.

Example training command:
```bash
python train_direct.py --config configs/unified_config.yaml
```

*Note: You may need to adjust the arguments based on the specific requirements of your task. Check the script's argument parser for more details.*

### 3. Running Inference

For running inference with a trained model, use `Inference_Engine.py` or `onnx_infer.py` for ONNX models.

-   **PyTorch Inference:** `Inference_Engine.py`
-   **ONNX Inference:** `onnx_infer.py` (after exporting the model using `ONNX_Export.py`)

## Testing and Validation

The primary script for testing and validating the model is `TEst and review/batch_evaluate.py`. This script evaluates a trained model against a dataset and computes performance metrics.

### How to Run Evaluation

1.  **Prepare your data:**
    -   You need a folder containing images and corresponding `.json` files with ground truth tags. The script expects each JSON file to have a `filename` key pointing to the image file and a `tags` key with a space-separated list of tags.

2.  **Run the evaluation script:**
    -   Modify the `main` function in `TEst and review/batch_evaluate.py` to point to your data folder, model path, and desired output directory.
    -   Execute the script:
        ```bash
        python "TEst and review/batch_evaluate.py"
        ```

3.  **Review the results:**
    -   The script will generate a summary JSON file with aggregate metrics (precision, recall, F1-score) and detailed per-tag performance. It also creates a `.jsonl` file with results for each individual image.

### Other Tools

-   `TEst and review/live_viewer.py`: Likely for interactive, real-time model evaluation.
-   `TEst and review/visualize_results.py`: Can be used to create visualizations from the evaluation results.
