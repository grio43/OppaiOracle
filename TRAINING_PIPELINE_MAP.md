# Training Pipeline Map

This document outlines the core components of the training pipeline for the OppaiOracle project. It focuses strictly on files involved in the training process, excluding supporting tools or analysis scripts.

## 1. Entry Points
*   **`train_direct.py`**: The main entry point for the training process. It orchestrates the entire pipeline: loading config, initializing the model and dataset, setting up the optimizer/scheduler, and running the training loop.
*   **`Start_AI_Training.ps1`**: A PowerShell wrapper script that sets up the environment (using `payton_env.ps1`) and launches `train_direct.py` with the correct arguments.

## 2. Configuration System
*   **`Configuration_System.py`**: Handles loading, validation, and access to all configuration settings. It defines dataclasses for different config sections (Model, Data, Training, etc.) and loads from `configs/unified_config.yaml`.
*   **`configs/unified_config.yaml`**: Single source of truth for model, data, training, inference, and export settings.

## 3. Data Pipeline
*   **`dataset_loader.py`**: The core data loading module.
    *   **`DatasetLoader`**: A standard PyTorch Dataset for loading images and JSON annotations.
    *   **`SidecarJsonDataset`**: A specialized dataset implementation that works with sidecar JSON files (primary method).
    *   **`DataLoader`**: A custom wrapper around PyTorch's DataLoader to handle specific threading/worker edge cases.
    *   **`create_dataloaders`**: Factory for train/val loaders and vocab creation.
    *   **`AugmentationStats` / `validate_dataset`**: Augmentation counters and dataset sanity checks.
*   **`shared_vocabulary.py`**: Shared vocabulary manager to reduce memory duplication across workers.
*   **`vocabulary.py`**: Manages the tag vocabulary.
    *   **`TagVocabulary`**: Handles bidirectional mapping between tag strings and integer indices.
    *   **`create_vocabulary_from_datasets`**: Scans the dataset to build the vocabulary file (`vocabulary.json`).
*   **`orientation_handler.py`**: (Imported by `dataset_loader.py` and `train_direct.py`) Manages image orientation logic for flips and directional tags.
    *   **`configs/orientation_map.json`**: Left/right mappings and flip rules consumed by the handler.

## 4. Model Architecture
*   **`model_architecture.py`**: Defines the neural network structure.
    *   **`create_model`**: Factory to construct the configured architecture.
    *   **`SimplifiedTagger`**: Vision Transformer (ViT) model optimized for multi-label tagging.
    *   **`SwinV2Tagger`**: Swin V2-based model variant for tagging.
    *   **`VisionTransformerConfig`**: Configuration dataclass for the model dimensions and hyperparameters.
    *   **`TransformerBlock`**: Implementation of a single ViT block (Attention + MLP).

## 5. Training Logic & Utilities
*   **`loss_functions.py`**: Defines the loss functions used for training.
    *   **`AsymmetricFocalLoss`**: The primary loss function for multi-label classification, designed to handle class imbalance (many negatives, few positives).
    *   **`MultiTaskLoss`**: Combines the tag loss and rating loss into a single scalar value.
*   **`training_utils.py`**: A collection of helpers for the training loop.
    *   **`CheckpointManager`**: Manages saving and loading model checkpoints (last, best, periodic).
    *   **`TrainingState`**: Persistent state tracking for resuming training.
    *   **`AsyncCheckpointWriter`**: Handles saving checkpoints in a background thread to prevent training stalls.
    *   **`MixedPrecisionTrainer`**: Utilities for handling bfloat16/float16 mixed-precision training.
    *   **`CosineAnnealingWarmupRestarts`**: A custom learning rate scheduler.
    *   **`EarlyStopping`**: Logic to stop training when validation performance plateaus.
    *   **`TrainingUtils`**: Optimizer/scheduler selection and parameter group setup.
*   **`adan_optimizer.py`**: Custom Adan optimizer implementation used by `TrainingUtils` when configured.
*   **`schedulers.py`**: Contains learning rate scheduler implementations.
    *   **`LinearWarmupCosineLR`**: A scheduler compatible with PyTorch 2.x that implements linear warmup followed by cosine annealing.
*   **`Monitor_log.py`**: Handles training monitoring and logging.
    *   **`TrainingMonitor`**: The main class for tracking metrics, managing TensorBoard/WandB writers, and logging system stats (CPU/GPU/RAM).
    *   **`ThreadSafeMetricsTracker`**: Ensures metric logging is thread-safe.

## 6. Evaluation & Metrics
*   **`evaluation_metrics.py`**: Computes performance metrics during validation.
    *   **`MetricComputer`**: Calculates Macro/Micro F1 scores and Mean Average Precision (mAP) for multi-label predictions.
*   **`validation_loop.py`**: A standalone script/module for running comprehensive validation (e.g., hierarchical metrics, specific tag analysis). `train_direct.py` has its own inline validation loop, but this module offers more advanced evaluation modes.

## 7. Legacy / Deprecated
*   Legacy Lightning entrypoints and pre-unified config helpers were removed during housekeeping.
*   Use `train_direct.py`, `training_utils.CheckpointManager`, and `training_config.py`.
