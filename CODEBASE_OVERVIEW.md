# Codebase Overview

This document provides an overview of the codebase, separating files into two categories: Pipeline Files and Standalone Utility Files.

## Pipeline Files

These files are core components of the machine learning pipeline, responsible for data loading, preprocessing, training, and inference.

-   `dataset_loader.py`: Handles loading the dataset.
-   `dataset_preprocessor.py`: Preprocesses the data before training.
-   `model_architecture.py`: Defines the neural network architecture.
-   `lightning_module.py`: Encapsulates the model, optimizer, and training logic for PyTorch Lightning.
-   `train_direct.py`: A script for training the model directly.
-   `train_lightning.py`: A script for training the model using PyTorch Lightning.
-   `Inference_Engine.py`: Handles model inference.
-   `validation_loop.py`: Contains the logic for the validation loop during training.
-   `loss_functions.py`: Defines custom loss functions.
-   `schedulers.py`: Contains learning rate schedulers.
-   `optimizer_config.py`: Configuration for optimizers.
-   `scheduler_config.py`: Configuration for schedulers.
-   `adan_optimizer.py`: Implementation of the Adan optimizer.
-   `training_utils.py`: Utility functions for training.
-   `Configuration_System.py`: Manages configuration for the pipeline.
-   `unified_training_config.py`: A unified configuration for training.
-   `ONNX_Export.py`: Exports the model to ONNX format.
-   `onnx_infer.py`: Performs inference with an ONNX model.
-   `evaluation_metrics.py`: Defines metrics for model evaluation.
-   `safe_checkpoint.py`: Handles saving and loading of model checkpoints.
-   `l1_cache.py`: L1 cache implementation.
-   `l2_cache.py`: L2 cache implementation.
-   `l2_cache_warmup.py`: A script to warm up the L2 cache.
-   `cache_codec.py`: Codec for caching.
-   `gpu_batch_processor.py`: Processes batches on the GPU.
-   `orientation_handler.py`: Handles image orientation.
-   `mask_utils.py`: Utilities for handling masks.
-   `model_metadata.py`: Handles model metadata.
-   `shared_vocabulary.py`: Manages a shared vocabulary.
-   `vocabulary.py`: Vocabulary management.
-   `vocab_utils.py`: Utilities for vocabulary management.
-   `schemas.py`: Data schemas.
-   `custom_drop_path.py`: Custom implementation of DropPath.
-   `utils/`: Directory for various utility modules.
    -   `cache_keys.py`: Defines cache keys.
    -   `cache_monitor.py`: Monitors the cache.
    -   `file_handlers.py`: Handlers for different file types.
    -   `logging_sanitize.py`: Sanitizes logs.
    -   `logging_setup.py`: Sets up logging.
    -   `metadata_ingestion.py`: Ingests metadata.
    -   `path_utils.py`: Utilities for handling paths.

## Standalone Utility Files

These files are standalone scripts for various tasks like analysis, debugging, and data exploration.

-   `analyze_cache_dtype.py`: Analyzes the data types in the cache.
-   `analyze_issues.py`: A script to analyze issues.
-   `analyze_tag_counts.py`: Analyzes the distribution of tags.
-   `auto_training_setup.py`: Automatically sets up the training environment.
-   `benchmark_vocab_load.py`: Benchmarks vocabulary loading speed.
-   `check_bf16_support.py`: Checks for bfloat16 support.
-   `check_for_lost_files.py`: Checks for lost files in the dataset.
-   `copy_test_files.py`: Copies test files.
-   `count_dataset.py`: Counts the number of items in the dataset.
-   `count_files.py`: Counts files in a directory.
-   `create_huge_png.py`: Creates a huge PNG file for testing.
-   `create_large_png.py`: Creates a large PNG file for testing.
-   `Dataset_Analysis.py`: A script for analyzing the dataset.
-   `demo_auto_config.py`: A demo for automatic configuration.
-   `downsample_gpu_accelerated.py`: Downsamples images using GPU acceleration.
-   `downsample_images_nas.py`: Downsamples images.
-   `example_optimizer_configs.py`: Example optimizer configurations.
-   `filter_low_tag_images_turbo.py`: A faster script to filter images with few tags.
-   `filter_low_tag_images.py`: Filters images with a low number of tags.
-   `manual_png_test.py`: A script for manually testing PNG files.
-   `Monitor_log.py`: Monitors a log file.
-   `run_test_downsample.py`: Runs a test for downsampling.
-   `vocab_append.py`: Appends to the vocabulary.
-   `scripts/convert_vocab_to_metadata.py`: Converts vocabulary to metadata.
-   `TEst and review/`: Directory for testing and reviewing results.
    -   `batch_evaluate.py`: Evaluates the model on a batch of data.
    -   `live_viewer.py`: A live viewer for results.
    -   `visualize_results.py`: Visualizes the results.
-   `tools/`: Directory for various tools.
    -   `calibrate_thresholds.py`: Calibrates thresholds for classification.
    -   `generate_manifests.py`: Generates manifests for the dataset.
