# Training Pipeline Map

This document outlines the core components of the training pipeline for the OppaiOracle project. It focuses strictly on files involved in the training process, excluding supporting tools or analysis scripts.

## 1. Entry Points
*   **`train_direct.py`**: The main entry point for the training process. It orchestrates the entire pipeline: loading config, initializing the model and dataset, setting up the optimizer/scheduler, and running the training loop. It replaces the deprecated `train_lightning.py`.
*   **`Start_AI_Training.ps1`**: A PowerShell wrapper script that sets up the environment (using `payton_env.ps1`) and launches `train_direct.py` with the correct arguments.

## 2. Configuration System
*   **`Configuration_System.py`**: Handles loading, validation, and access to all configuration settings. It defines dataclasses for different config sections (Model, Data, Training, etc.) and loads from `configs/unified_config.yaml`.
*   **`unified_training_config.py`** (Implicit): While not explicitly read in the trace, this likely defines the schema or default values used by `Configuration_System.py`.

## 3. Data Pipeline
*   **`dataset_loader.py`**: The core data loading module.
    *   **`DatasetLoader`**: A standard PyTorch Dataset for loading images and JSON annotations.
    *   **`SidecarJsonDataset`**: A specialized dataset implementation that works with sidecar JSON files (primary method).
    *   **`ImagePreFetcher`**: Handles background loading and processing of images to hide I/O latency.
    *   **`DataLoader`**: A custom wrapper around PyTorch's DataLoader to handle specific threading/worker edge cases.
*   **`vocabulary.py`**: Manages the tag vocabulary.
    *   **`TagVocabulary`**: Handles bidirectional mapping between tag strings and integer indices.
    *   **`create_vocabulary_from_datasets`**: Scans the dataset to build the vocabulary file (`vocabulary.json`).
*   **`orientation_handler.py`**: (Imported by `dataset_loader.py` and `train_direct.py`) Manages image orientation logic, likely for handling rotation/flipping augmentations correctly with respect to tags (e.g., swapping "left" and "right" tags).

## 4. Model Architecture
*   **`model_architecture.py`**: Defines the neural network structure.
    *   **`SimplifiedTagger`**: The main model class, a Vision Transformer (ViT) optimized for multi-label tagging.
    *   **`VisionTransformerConfig`**: Configuration dataclass for the model dimensions and hyperparameters.
    *   **`TransformerBlock`**: Implementation of a single ViT block (Attention + MLP).

## 5. Training Logic & Utilities
*   **`loss_functions.py`**: Defines the loss functions used for training.
    *   **`AsymmetricFocalLoss`**: The primary loss function for multi-label classification, designed to handle class imbalance (many negatives, few positives).
    *   **`MultiTaskLoss`**: Combines the tag loss and rating loss into a single scalar value.
*   **`training_utils.py`**: A collection of helpers for the training loop.
    *   **`CheckpointManager`**: Manages saving and loading model checkpoints (last, best, periodic).
    *   **`AsyncCheckpointWriter`**: Handles saving checkpoints in a background thread to prevent training stalls.
    *   **`MixedPrecisionTrainer`**: Utilities for handling bfloat16/float16 mixed-precision training.
    *   **`CosineAnnealingWarmupRestarts`**: A custom learning rate scheduler.
    *   **`EarlyStopping`**: Logic to stop training when validation performance plateaus.
*   **`schedulers.py`**: Contains learning rate scheduler implementations.
    *   **`LinearWarmupCosineLR`**: A scheduler compatible with PyTorch 2.x that implements linear warmup followed by cosine annealing.
*   **`Monitor_log.py`**: Handles training monitoring and logging.
    *   **`TrainingMonitor`**: The main class for tracking metrics, managing TensorBoard/WandB writers, and logging system stats (CPU/GPU/RAM).
    *   **`ThreadSafeMetricsTracker`**: Ensures metric logging is thread-safe.
*   **`safe_checkpoint.py`** (Deprecated): A legacy wrapper for checkpoint loading. Functionality has been moved to `CheckpointManager` in `training_utils.py`.

## 6. Evaluation & Metrics
*   **`evaluation_metrics.py`**: Computes performance metrics during validation.
    *   **`MetricComputer`**: Calculates Macro/Micro F1 scores and Mean Average Precision (mAP) for multi-label predictions.
*   **`validation_loop.py`**: A standalone script/module for running comprehensive validation (e.g., hierarchical metrics, specific tag analysis). `train_direct.py` has its own inline validation loop, but this module offers more advanced evaluation modes.