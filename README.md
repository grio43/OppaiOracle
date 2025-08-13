# Anime Image Tagger - Deep Learning Multi-Label Classification System

A production-ready Vision Transformer-based system for automated anime/manga image tagging with support for 100,000+ tags. This project implements a comprehensive training pipeline with orientation-aware augmentation, hierarchical tag prediction, and optimized inference.

## 📋 Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Architecture](#architecture)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Dataset Preparation](#dataset-preparation)
- [Training](#training)
- [Inference](#inference)
- [Evaluation](#evaluation)
- [Model Export](#model-export)
- [Configuration](#configuration)
- [API Reference](#api-reference)
- [Performance Metrics](#performance-metrics)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

## 🎯 Overview

This project provides a complete solution for multi-label classification of anime/manga images, capable of predicting from a vocabulary of 100,000+ tags. The system is designed for production deployment with features like ONNX export, distributed training, and comprehensive monitoring.

### Core Capabilities

- **Multi-label classification** with 100,000+ tags
- **Hierarchical prediction** with 20 groups of 10,000 tags each
- **Content rating classification** (general, sensitive, questionable, explicit)
- **High accuracy** with asymmetric focal loss optimization
- **Production-ready inference** with batch processing and API support

## ✨ Key Features

### Model Architecture
- **Vision Transformer (ViT)** backbone with 1.28B parameters
- Patch size: 16x16 for 640x640 input images
- 24 transformer layers with 16 attention heads
- Gradient checkpointing for memory efficiency
- Flash attention support for faster training

### Training Pipeline
- **Orientation-aware augmentation** with comprehensive left/right tag mapping
- **Asymmetric focal loss** optimized for multi-label classification
- **Mixed precision training** (FP16/BF16) support
- **Distributed training** with DDP
- **Curriculum learning** strategy
- **Automatic checkpointing** and recovery

### Data Handling
- **HDF5-based data loading** for efficient I/O
- **Frequency-weighted sampling** for balanced training
- **Smart caching** with configurable memory limits
- **Stratified dataset splitting**
- **Comprehensive data analysis tools**

### Inference & Deployment
- **Optimized inference engine** with correct normalization
- **ONNX export** with mobile/edge optimization
- **REST API** with FastAPI
- **Batch processing** capabilities
- **TensorRT acceleration** support

### Monitoring & Logging
- **Real-time training monitoring** with TensorBoard/W&B
- **System resource tracking** (CPU, GPU, memory)
- **Alert system** for training anomalies
- **Comprehensive metrics tracking**
- **Performance profiling** tools

## 🏗️ Architecture

```
Project Structure:
├── model_architecture.py    # Vision Transformer model definition
├── train_direct.py          # Main training script with orientation handling
├── HDF5_loader.py          # Enhanced data loader with augmentation
├── loss_functions.py       # Asymmetric focal loss implementation
├── orientation_handler.py   # Orientation-aware tag mapping
├── tag_vocabulary.py       # Vocabulary management and data preparation
├── Inference_Engine.py     # Production inference with fixed normalization
├── Evaluation_Metrics.py   # Comprehensive evaluation metrics
├── ONNX_Export.py         # Model export for deployment
├── Monitor_log.py         # Training monitoring and logging
├── Configuration_System.py # Centralized configuration management
├── Dataset_Analysis.py    # Dataset analysis and validation tools
├── training_utils.py      # Training utilities and helpers
├── validation_loop.py     # Validation pipeline
└── configs/
    ├── training_config.yaml
    ├── inference_config.yaml
    ├── export_config.yaml
    └── orientation_map.json
```

## 📦 Installation

### Requirements
- Python 3.8+
- CUDA 11.3+ (for GPU support)
- 16GB+ VRAM recommended for training
- 100GB+ disk space for datasets

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/anime-image-tagger.git
cd anime-image-tagger

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

### Dependencies

Create `requirements.txt`:
```txt
numpy>=1.21.0
pandas>=1.3.0
pillow>=9.0.0
opencv-python>=4.5.0
scikit-learn>=1.0.0
matplotlib>=3.4.0
seaborn>=0.11.0
tqdm>=4.62.0
pyyaml>=5.4.0
h5py>=3.6.0
tensorboard>=2.8.0
wandb>=0.12.0  # optional
fastapi>=0.70.0  # for API
uvicorn>=0.15.0  # for API
python-multipart>=0.0.5  # for API
onnx>=1.11.0
onnxruntime>=1.10.0
psutil>=5.8.0
GPUtil>=1.4.0  # optional
prometheus-client>=0.12.0  # optional
imagehash>=4.2.0  # optional
wordcloud>=1.8.0  # optional
```

## 🚀 Quick Start

### 1. Prepare Configuration

```yaml
# configs/training_config.yaml
num_epochs: 100
learning_rate: 1.0e-4
batch_size: 32
device: cuda
use_amp: true
random_flip_prob: 0.2  # Enable orientation-aware flips
```

### 2. Prepare Dataset

```python
# Prepare vocabulary and dataset
python tag_vocabulary.py \
    --metadata_dir /path/to/json/metadata \
    --vocab_path ./vocabulary \
    --output_dir ./processed_data \
    --phase1_size 4000000 \
    --total_size 8500000
```

### 3. Train Model

```python
# Start training with orientation handling
python train_direct.py \
    --data_dir ./data/images \
    --json_dir ./data/annotations \
    --vocab_path ./vocabulary \
    --config configs/training_config.yaml
```

### 4. Run Inference

```python
# Single image inference
python Inference_Engine.py \
    --model ./checkpoints/best_model.pt \
    --vocab ./vocabulary \
    --image ./test_image.jpg \
    --threshold 0.5
```

## 📊 Dataset Preparation

### Dataset Structure

```
dataset/
├── images/
│   ├── image_001.jpg
│   ├── image_002.png
│   └── ...
├── annotations/
│   ├── metadata_001.json
│   ├── metadata_002.json
│   └── ...
└── vocabulary/
    ├── tags.txt
    └── vocabulary.pkl
```

### Annotation Format

```json
{
  "filename": "image_001.jpg",
  "tags": ["1girl", "solo", "long_hair", "blue_eyes", "smile"],
  "rating": "general",
  "quality_score": 8.5
}
```

### Data Analysis

```bash
# Analyze dataset quality and statistics
python Dataset_Analysis.py \
    ./dataset/images \
    --output-dir ./analysis \
    --sample-size 10000 \
    --report-format html
```

## 🎓 Training

### Basic Training

```python
from train_direct import train_with_orientation_tracking
from pathlib import Path

# Configure and start training
config = {
    "data_dir": Path("./data/images"),
    "json_dir": Path("./data/annotations"),
    "vocab_path": Path("./vocabulary"),
    "num_epochs": 100,
    "batch_size": 32,
    "learning_rate": 1e-4,
    "random_flip_prob": 0.2,  # Enable orientation-aware augmentation
    "orientation_map_path": Path("configs/orientation_map.json")
}

train_with_orientation_tracking(config)
```

### Distributed Training

```bash
# Multi-GPU training with DDP
python -m torch.distributed.launch \
    --nproc_per_node=4 \
    train_direct.py \
    --distributed \
    --config configs/training_config.yaml
```

### Training Monitoring

The system provides comprehensive monitoring:

```python
from Monitor_log import TrainingMonitor, MonitorConfig

config = MonitorConfig(
    use_tensorboard=True,
    use_wandb=False,
    track_gpu_metrics=True,
    enable_alerts=True,
    alert_on_nan_loss=True,
    alert_on_gpu_memory_threshold=0.9
)

monitor = TrainingMonitor(config)

# In training loop
monitor.log_step(
    step=global_step,
    loss=loss_value,
    metrics={'accuracy': acc, 'f1': f1},
    learning_rate=lr
)
```

## 🔮 Inference

### Python API

```python
from Inference_Engine import InferenceEngine, InferenceConfig

# Configure inference
config = InferenceConfig(
    model_path="./checkpoints/best_model.pt",
    vocab_dir="./vocabulary",
    device="cuda",
    use_fp16=True,
    prediction_threshold=0.5,
    adaptive_threshold=True,
    min_predictions=5,
    max_predictions=50,
    # CRITICAL: Use correct normalization for anime models
    normalize_mean=(0.5, 0.5, 0.5),
    normalize_std=(0.5, 0.5, 0.5)
)

# Initialize engine
engine = InferenceEngine(config)

# Single image prediction
result = engine.predict("./anime_image.jpg")
print(f"Predicted tags: {result['tags']}")
print(f"Inference time: {result['inference_time']}ms")

# Batch prediction
results = engine.predict_batch(["img1.jpg", "img2.jpg", "img3.jpg"])
```

### REST API

```bash
# Start API server
python Inference_Engine.py \
    --model ./checkpoints/best_model.pt \
    --vocab ./vocabulary \
    --api \
    --api-port 8000
```

API endpoints:
- `POST /predict` - Upload image for prediction
- `POST /predict_url` - Predict from image URL
- `GET /stats` - Get inference statistics
- `GET /health` - Health check

### Batch Processing

```python
from Inference_Engine import BatchInferenceProcessor

processor = BatchInferenceProcessor(engine, num_workers=4)

# Process entire directory
results = processor.process_directory(
    directory="./images_to_tag",
    output_file="./results.json",
    extensions=['.jpg', '.png', '.webp']
)
```

## 📈 Evaluation

### Validation Pipeline

```bash
# Full validation with all metrics
python validation_loop.py \
    --checkpoint ./checkpoints/best_model.pt \
    --mode full \
    --batch-size 64 \
    --output-dir ./validation_results \
    --create-plots
```

### Evaluation Metrics

The system computes comprehensive metrics:

- **Micro/Macro Precision, Recall, F1**
- **Mean Average Precision (mAP)**
- **Top-k Accuracy** (k=1,5,10,20,50)
- **Hamming Loss**
- **Per-tag Performance Analysis**
- **Frequency-based Metrics**
- **Hierarchical Group Metrics**

```python
from Evaluation_Metrics import evaluate_model, MetricConfig

config = MetricConfig(
    compute_per_tag_metrics=True,
    compute_auc=True,
    top_k_values=[1, 5, 10, 20, 50],
    frequency_bins=[10, 100, 1000, 10000]
)

metrics = evaluate_model(
    model=model,
    dataloader=val_loader,
    config=config,
    device=device,
    tag_names=vocab.tags,
    save_plots=True
)

print(f"mAP: {metrics['mAP']:.4f}")
print(f"F1 Micro: {metrics['f1_micro']:.4f}")
```

## 📤 Model Export

### ONNX Export

```bash
# Export to ONNX with optimizations
python ONNX_Export.py \
    ./checkpoints/best_model.pt \
    ./vocabulary \
    --output model.onnx \
    --variants full mobile quantized \
    --optimize \
    --benchmark
```

### Export Variants

1. **Full Model** - Maximum accuracy, FP32
2. **Mobile Model** - Optimized for edge devices
3. **Quantized Model** - INT8 quantization for speed

### TensorRT Optimization

```python
from ONNX_Export import ONNXExporter, ONNXExportConfig

config = ONNXExportConfig(
    checkpoint_path="./checkpoints/best_model.pt",
    vocab_dir="./vocabulary",
    export_format="tensorrt",
    fp16_mode=True,
    max_batch_size=32
)

exporter = ONNXExporter(config)
results = exporter.export()
```

## ⚙️ Configuration

### Configuration System

The project uses a hierarchical configuration system:

```python
from Configuration_System import ConfigManager, FullConfig

# Load configuration
manager = ConfigManager()
config = manager.load_from_file("configs/full_config.yaml")

# Override with environment variables
# ANIME_TAGGER_TRAINING__LEARNING_RATE=0.0005
manager.update_from_env()

# Override with command line arguments
manager.update_from_args(args)

# Save final configuration
manager.save_to_file("final_config.yaml")
```

### Key Configuration Files

1. **training_config.yaml** - Training hyperparameters
2. **inference_config.yaml** - Inference settings
3. **orientation_map.json** - Tag orientation mappings
4. **export_config.yaml** - Model export configuration

### Orientation Mapping

Critical for correct augmentation:

```json
{
  "explicit_mappings": {
    "hair_over_left_eye": "hair_over_right_eye",
    "facing_left": "facing_right",
    "hand_on_left_hip": "hand_on_right_hip"
  },
  "symmetric_tags": ["standing", "sitting", "looking_at_viewer"],
  "skip_flip_tags": ["text", "signature", "watermark"]
}
```

## 📊 Performance Metrics

### Model Performance

| Metric | Value | Notes |
|--------|-------|-------|
| mAP | TBD | Mean Average Precision |
| F1 Micro | TBD | Overall F1 score |
| Precision@5 | TBD | Top-5 precision |
| Recall@20 | TBD | Top-20 recall |
| Inference Speed | TBD | Per image on V100 |
| Model Size | TBD params | ~5GB FP32 |

### Training Performance

| Configuration | Speed | Memory |
|--------------|-------|--------|
| Single V100 | TBD img/s | 14GB |
| 4x V100 DDP | TBD img/s | 14GB/GPU |
| Mixed Precision | TBD img/s | 9GB |
| Gradient Checkpointing | TBD img/s | 8GB |

## 🔧 Troubleshooting

### Common Issues

#### 1. Normalization Mismatch
```python
# WARNING: Config appears to use ImageNet normalization!
# FIX: Use anime-optimized values
normalize_mean = (0.5, 0.5, 0.5)
normalize_std = (0.5, 0.5, 0.5)
```

#### 2. CUDA Out of Memory
```python
# Solutions:
# 1. Enable gradient checkpointing
model_config.gradient_checkpointing = True

# 2. Reduce batch size
config.batch_size = 16

# 3. Use mixed precision
config.use_amp = True
```

#### 3. Orientation Tag Errors
```python
# Check unmapped tags after training
with open("unmapped_orientation_tags.json") as f:
    unmapped = json.load(f)
# Add to orientation_map.json
```

### Performance Optimization

1. **Enable Flash Attention** for 2x training speedup
2. **Use Mixed Precision** for memory efficiency
3. **Enable Gradient Checkpointing** for large batches
4. **Use TensorRT** for inference acceleration
5. **Implement Smart Caching** for data loading

## 🤝 Contributing

We welcome contributions! Please follow these guidelines:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Setup

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
pytest tests/

# Check code style
flake8 .
black --check .

# Run type checking
mypy .
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Vision Transformer architecture based on [Google Research](https://github.com/google-research/vision_transformer)
- Asymmetric Focal Loss inspired by [ASL paper](https://arxiv.org/abs/2009.14119)
- Dataset preparation tools adapted from Danbooru tagging research
- [DeepGHS](https://huggingface.co/deepghs) and the artists on his datasets
  
---

**Note**: This project is designed for research and educational purposes. Please ensure you have appropriate rights to use any datasets and respect content creators' licenses.
