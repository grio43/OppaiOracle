# MAID: Model for AI-based Detection

This repository contains the code for MAID, a deep learning model for AI-based detection. The project is built using PyTorch and includes tools for training, evaluation, and deployment.

## Project Structure

The repository is organized as follows:

-   `configs/`: Contains configuration files for the model and training process.
-   `data/`: (Assumed) Should contain the training and evaluation data.
-   `logs/`: Contains logs from training and other processes.
-   `scripts/`: Contains various utility scripts.
-   `tools/`: Contains tools for calibration and other tasks.
-   `TEst and review/`: Contains scripts for testing and visualizing results.
-   `utils/`: Contains utility functions used across the project.

## Core Components

-   **`model_architecture.py`**: Defines the neural network architecture.
-   **`train_direct.py`**: The main script for training the model.
-   **`Inference_Engine.py`**: Handles model inference.
-   **`ONNX_Export.py`**: Exports the trained model to ONNX format for optimized inference.
-   **`HDF5_loader.py`**: Loads data from HDF5 files.
-   **`validation_loop.py`**: Contains the logic for the model validation loop.

## Getting Started

### Prerequisites

The project dependencies are listed in `requirements.txt`. You can install them using pip:

```bash
pip install -r requirements.txt
```

### Training

To train the model, you can run the `train_direct.py` script:

```bash
python train_direct.py --config configs/unified_config.yaml
```

(Note: The exact command might vary depending on the required arguments.)

### Inference

The `Inference_Engine.py` and `onnx_infer.py` can be used for running inference with the trained model.

## API

The project includes an optional API for serving the model, built with FastAPI. To run the API, you can use `uvicorn`.
