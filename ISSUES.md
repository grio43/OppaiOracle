# Pipeline Issues and Priority List

This document outlines the identified issues in the codebase, prioritized by their potential impact on the project's stability, maintainability, and clarity.

## Priority: High

### 1. Inconsistent `num_tags` Handling in Inference Engine
- **Description**: The `Inference_Engine.py` uses complex and fragile logic to determine the number of tags (`num_tags`) for the model. It attempts to infer this value from multiple sources, including the model checkpoint, the vocabulary file, and even the state dictionary keys. This can easily lead to mismatches between the model architecture and the vocabulary, causing runtime errors or silent mispredictions. A single, reliable source of truth for `num_tags` should be established.
- **Affected File**: `Inference_Engine.py`
- **Code Snippet**:
  ```python
  # ... inside ModelWrapper.load_model ...
  if 'num_tags' not in vit_config_dict:
      if self.tag_names:
          vit_config_dict['num_tags'] = len(self.tag_names)
          logger.info(f"Setting num_tags={len(self.tag_names)} from tag_names")
      elif 'num_classes' in meta:
          vit_config_dict['num_tags'] = meta['num_classes']
          logger.info(f"Setting num_tags={meta['num_classes']} from checkpoint")
      else:
          # Try to infer from tag_head dimensions in state_dict
          for key, value in state_dict.items():
              if 'tag_head.weight' in key:
                  vit_config_dict['num_tags'] = value.shape[0]
                  # ...
                  break
          else:
              raise ValueError("Cannot determine num_tags from checkpoint or config")
  ```

### 2. Fragmented and Overlapping Configuration Management
- **Description**: The project suffers from a scattered configuration system. There are multiple files (`Configuration_System.py`, `optimizer_config.py`, `scheduler_config.py`, `unified_training_config.py`) that manage different parts of the configuration. This fragmentation makes it difficult to understand the overall configuration, track down settings, and ensure consistency. It also increases the risk of conflicts and makes maintenance more challenging.
- **Affected Files**: `Configuration_System.py`, `optimizer_config.py`, `scheduler_config.py`, `unified_training_config.py`
- **Code Snippets**:
  ```python
  # optimizer_config.py
  @dataclass
  class AdamW8bitConfig:
      """Configuration for AdamW8bit optimizer with dataset-aware scaling."""
      # ...

  # scheduler_config.py
  @dataclass
  class SchedulerConfig:
      """Configuration for learning rate schedulers."""
      # ...
  ```

### 3. Brittle and Hardcoded Paths in Inference Engine
- **Description**: The `Inference_Engine.py` contains hardcoded fallback logic for locating critical files like the vocabulary and orientation map. This makes the inference process brittle and difficult to configure, as it relies on a specific directory structure. This should be replaced with a clear and explicit configuration-driven approach.
- **Affected File**: `Inference_Engine.py`
- **Code Snippet**:
  ```python
  def _load_vocab_path() -> Path:
      """Resolve vocabulary path via unified_config.yaml, with sensible fallbacks."""
      try:
          cfg = yaml.safe_load((PROJECT_ROOT / "configs" / "unified_config.yaml").read_text(encoding="utf-8")) or {}
      except FileNotFoundError:
          logger.warning("unified_config.yaml not found, using default vocabulary path")
          cfg = {}
      # ... more complex fallback logic ...
  ```

### 4. Unreliable ONNX Export Process
- **Description**: The ONNX export logic in `ONNX_Export.py` is overly complex, especially around metadata and vocabulary embedding. It has multiple fallbacks and checks that can lead to silent failures, resulting in ONNX models that are incomplete or not self-contained. This makes deployment unreliable.
- **Affected File**: `ONNX_Export.py`
- **Code Snippet**:
  ```python
  def _add_metadata(self, model_path: Path):
      """Add metadata to ONNX model"""
      try:
          model = onnx.load(str(model_path))
          # ... complex logic to embed vocabulary ...
          if not vocab_embedded_successfully:
              if self.config.require_embedded_vocabulary:
                  raise RuntimeError("No vocabulary available for embedding.")
              else:
                  logger.warning("Exporting model without embedded vocabulary.")
          # ...
      except Exception as e:
          logger.warning(f"Failed to add metadata: {e}")
  ```

## Priority: Medium

### 5. Obsolete HDF5 Pipeline
- **Description**: The `dataset_preprocessor.py` script is designed to create HDF5 datasets, but the `dataset_loader.py` explicitly states that this pipeline is obsolete and has been replaced by on-the-fly JSON loading. The presence of this dead code is confusing and adds unnecessary clutter to the codebase.
- **Affected Files**: `dataset_preprocessor.py`, `dataset_loader.py`
- **Code Snippet** (from `dataset_loader.py`):
  ```python
  """
  ...
  ⚠️ IMPORTANT - OBSOLETE CODE WARNING:
  ======================================
  The file `dataset_preprocessor.py` (formerly `tag_vocabulary.py`) creates HDF5 files
  (training_data.h5, tag_indices.json, splits.json) that are NOT used by this system.
  ...
  DO NOT USE dataset_preprocessor.py - it is dead code maintained only for
  historical reference.
  ...
  """
  ```

### 6. Dual Training Scripts
- **Description**: The project contains two separate training scripts: `train_direct.py` and `train_lightning.py`. While `train_direct.py` appears to be more feature-complete and actively used, maintaining two scripts for the same purpose leads to duplicated effort and potential inconsistencies. It would be better to consolidate into a single, well-maintained training script.
- **Affected Files**: `train_direct.py`, `train_lightning.py`

## Priority: Low

### 7. Unused `evaluation_metrics.py`
- **Description**: The `evaluation_metrics.py` file provides a `MetricComputer` class, but the primary training and validation scripts (`train_direct.py` and `validation_loop.py`) appear to use `torchmetrics` directly for metric calculations. This suggests that `evaluation_metrics.py` is either obsolete or not fully integrated.
- **Affected Files**: `evaluation_metrics.py`, `train_direct.py`, `validation_loop.py`
- **Code Snippet**:
  ```python
  # evaluation_metrics.py
  @dataclass
  class MetricComputer:
      """Compute macro/micro F1 and mAP for multilabel classification."""
      num_labels: int
      threshold: float = 0.5
      # ...
  ```

### 8. Redundant `safe_checkpoint.py`
- **Description**: The `safe_checkpoint.py` script provides a basic function for loading model checkpoints. However, `training_utils.py` contains a more comprehensive `CheckpointManager` class that handles both saving and loading of checkpoints with more features. The role of `safe_checkpoint.py` is unclear and it is likely redundant.
- **Affected Files**: `safe_checkpoint.py`, `training_utils.py`
- **Code Snippet**:
  ```python
  # safe_checkpoint.py
  def safe_load_checkpoint(
      path: Union[str, Path],
      validate_values: bool = True,
      allow_nan: bool = False
  ) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
      # ...
  ```

### 9. Legacy `DatasetLoader` Class
- **Description**: The `dataset_loader.py` file contains a class named `DatasetLoader` which, according to its own docstring, is a legacy component from the old HDF5 pipeline. Its presence alongside the active `SidecarJsonDataset` class is confusing.
- **Affected File**: `dataset_loader.py`
- **Code Snippet**:
  ```python
  class DatasetLoader(Dataset):
      def __init__(
          self,
          annotations_path,
          image_dir,
          # ...
      ):
          """
          Dataset loader for images and JSON metadata.
          Note: Despite legacy naming, this does NOT handle HDF5 files.
          """
          # ...
  ```
