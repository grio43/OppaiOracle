# Anime Image Tagger

A state-of-the-art Vision Transformer system for comprehensive anime image tagging with 200,000 hierarchically organized tags. This system combines advanced deep learning techniques with production-ready deployment capabilities.

## 🌟 Key Features

### Model Architecture
- **Large-scale Vision Transformer**: Scalable from 1B to 3B parameters
- **Hierarchical Tag Organization**: 200k tags organized into 20 groups of 10k each
- **Multi-Teacher Distillation**: Combines anime-specific and CLIP teachers
- **Progressive Scaling**: Automated model scaling with intelligent weight initialization

### Training Pipeline
- **Efficient HDF5 Data Pipeline**: Preprocessed teacher features for faster training
- **Mixed Precision Training**: FP16 support with automatic loss scaling
- **Distributed Training**: Multi-GPU and multi-node support
- **Comprehensive Monitoring**: Real-time metrics, system monitoring, and alerting

### Production Ready
- **Optimized Inference**: Batch processing, TensorRT optimization, quantization
- **Multiple Export Formats**: ONNX export with mobile optimization
- **REST API**: FastAPI-based inference server
- **Flexible Deployment**: CPU/GPU inference with adaptive batching

## 🏗️ Architecture Overview

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Raw Images    │ -> │ Feature Pipeline │ -> │ HDF5 Storage    │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                ↓
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│ Student Model   │ <- │ Training System  │ -> │ Validation      │
│ (1B-3B params)  │    │ (Multi-Teacher)  │    │ & Metrics       │
└─────────────────┘    └──────────────────┘    └─────────────────┘
        ↓
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│ ONNX Export     │    │ Inference Engine │    │ Production API  │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## 🚀 Quick Start

### Prerequisites
```bash
# Core dependencies
pip install torch torchvision transformers
pip install h5py pillow webp numpy pandas
pip install scikit-learn matplotlib seaborn tqdm
pip install fastapi uvicorn onnx onnxruntime

# Optional: for advanced features
pip install tensorboard wandb prometheus-client
pip install torch-tensorrt  # for TensorRT optimization
```

### 1. Data Preprocessing
```bash
# Extract teacher features from your dataset
python pretrain.py
```

### 2. Build Vocabulary
```bash
# Create tag vocabulary from your annotations
python tag_vocabulary.py --tag_files /path/to/tags/*.json \
                         --output_dir ./vocabulary \
                         --min_count 10 \
                         --max_tags 200000
```

### 3. Training
```python
from Configuration_System import load_config
from HDF5_loader import create_dataloaders
from model_architecture import create_model

# Load configuration
config = load_config("configs/training_config.yaml")

# Create data loaders
train_loader, val_loader = create_dataloaders(
    hdf5_dir="./teacher_features",
    vocab_dir="./vocabulary",
    batch_size=32
)

# Train model
model = create_model(config=config.model)
# Training loop implementation...
```

### 4. Inference
```python
from Inference_Engine import InferenceEngine, InferenceConfig

# Setup inference
config = InferenceConfig(
    model_path="./models/anime_tagger.pt",
    vocab_dir="./vocabulary",
    prediction_threshold=0.5
)

engine = InferenceEngine(config)

# Tag single image
result = engine.predict("path/to/image.jpg")
print(f"Tags: {[tag['tag'] for tag in result['tags']]}")

# Start API server
if __name__ == "__main__":
    from Inference_Engine import InferenceAPI
    api = InferenceAPI(engine, config)
    api.run()  # Starts on localhost:8000
```

## 📁 Project Structure

```
anime-image-tagger/
├── Configuration_System.py    # Centralized configuration management
├── Dataset_Analysis.py        # Dataset analysis and statistics tools
├── Evaluation_Metrics.py      # Comprehensive evaluation metrics
├── HDF5_loader.py             # Efficient data loading pipeline
├── Inference_Engine.py        # Production inference system
├── Monitor_log.py             # Training monitoring and logging
├── ONNX_Export.py             # Model export utilities
├── Progressive_model.py       # Model scaling (1B->3B parameters)
├── pretrain.py                # Teacher feature extraction
├── tag_vocabulary.py          # Tag vocabulary management
├── validation_loop.py         # Validation pipeline
└── configs/                   # Configuration files
    ├── training_config.yaml
    ├── inference_config.yaml
    └── export_config.yaml
```

## 🎯 Model Configurations

### Available Model Sizes
- **1B Parameters**: `hidden_size=1536, num_layers=28, num_heads=24`
- **1.5B Parameters**: `hidden_size=1792, num_layers=32, num_heads=28`
- **2B Parameters**: `hidden_size=2048, num_layers=36, num_heads=32`
- **3B Parameters**: `hidden_size=2304, num_layers=40, num_heads=36`

### Progressive Scaling
```python
from Progressive_model import scale_model_checkpoint

# Scale from 1B to 1.5B
scaled_model = scale_model_checkpoint(
    checkpoint_path="model_1B.pt",
    target_size="1.5B",
    method="depth_width"
)
```

## 📊 Training Features

### Multi-Teacher Distillation
- **Anime Teacher**: Specialized anime tagging model (70k tags)
- **CLIP Teacher**: Vision-language understanding
- **Distillation Loss**: Combines both teacher outputs

### Advanced Training Techniques
- **Curriculum Learning**: Progressive difficulty increase
- **Focal Loss**: Handles extreme label imbalance
- **Label Smoothing**: Prevents overconfident predictions
- **Gradient Clipping**: Stable training for large models

### Monitoring & Logging
- **TensorBoard**: Real-time training visualization
- **Weights & Biases**: Experiment tracking (optional)
- **Prometheus**: Production metrics collection
- **System Monitoring**: GPU/CPU/memory usage tracking

## 🔧 Configuration System

The system uses a sophisticated configuration management system:

```python
# Load from file with environment override support
config = load_config(
    config_file="configs/training.yaml",
    env_prefix="ANIME_TAGGER_"
)

# Environment variables override file settings
# ANIME_TAGGER_TRAINING__LEARNING_RATE=0.0001
# ANIME_TAGGER_MODEL__NUM_GROUPS=25
```

## 📈 Evaluation & Metrics

Comprehensive evaluation system with multiple metrics:

- **Multi-label Classification**: Precision, Recall, F1 (micro/macro/samples)
- **Hierarchical Metrics**: Per-group performance analysis  
- **Top-k Accuracy**: Performance at different k values
- **Threshold Analysis**: Optimal threshold detection
- **Tag Frequency Analysis**: Performance by tag popularity
- **Coverage Metrics**: Tag diversity and distribution

```python
from validation_loop import ValidationRunner, ValidationConfig

# Run comprehensive validation
config = ValidationConfig(
    checkpoint_path="./models/best_model.pt",
    mode="full",  # "full", "fast", "tags", "hierarchical"
    create_visualizations=True
)

runner = ValidationRunner(config)
results = runner.validate()
```

## 🌐 Production Deployment

### ONNX Export
```python
from ONNX_Export import ONNXExporter, ONNXExportConfig

config = ONNXExportConfig(
    checkpoint_path="./models/anime_tagger.pt",
    output_path="./exports/model.onnx",
    export_variants=["full", "mobile", "quantized"],
    optimize=True
)

exporter = ONNXExporter(config)
exporter.export()
```

### API Deployment
```bash
# Start inference API
python Inference_Engine.py --model ./models/anime_tagger.pt \
                           --vocab ./vocabulary \
                           --api \
                           --api-port 8000

# Test API
curl -X POST "http://localhost:8000/predict" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@image.jpg"
```

### Docker Deployment
```dockerfile
FROM nvidia/cuda:11.8-runtime-ubuntu20.04

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8000

CMD ["python", "Inference_Engine.py", "--api", "--api-port", "8000"]
```

## 🎨 Tag System

### Hierarchical Organization
- **20 Groups**: Balanced distribution of 200k tags
- **Smart Grouping**: Frequency-balanced, type-based, or co-occurrence
- **Special Categories**: Ratings, quality, artists, characters, copyright

### Tag Types
- **General Tags**: Descriptive content tags
- **Artist Tags**: Artist attribution
- **Character Tags**: Character identification  
- **Copyright Tags**: Series/franchise tags
- **Meta Tags**: Image quality/technical tags

## 📚 Advanced Features

### Dataset Analysis
```python
from Dataset_Analysis import DatasetAnalyzer, AnalysisConfig

config = AnalysisConfig(
    dataset_paths=["./datasets/anime_images"],
    analyze_duplicates=True,
    analyze_quality=True,
    create_visualizations=True
)

analyzer = DatasetAnalyzer(config)
stats = analyzer.analyze_dataset()
```

### Batch Processing
```python
from Inference_Engine import BatchInferenceProcessor

processor = BatchInferenceProcessor(engine, num_workers=4)
results = processor.process_directory(
    directory="./images",
    output_file="./results.json"
)
```

## 🔬 Research Features

- **Progressive Model Scaling**: Automated scaling with weight transfer
- **Advanced Metrics**: Comprehensive evaluation suite
- **Ablation Studies**: Component-wise performance analysis
- **Hyperparameter Optimization**: Automated tuning support

## 🤝 Contributing

This system is designed for research and production use in anime image analysis. Key areas for contribution:

- **Model Architectures**: New backbone architectures
- **Training Techniques**: Advanced training methods
- **Deployment**: New deployment backends and optimizations
- **Evaluation**: Additional metrics and analysis tools

## 📄 License

Please check the individual model licenses and dataset licenses when using this system.

## 📞 Contact

Creator: @Grio43 on Telegram

## 🙏 Acknowledgments

- Built on PyTorch and Transformers
- Uses Vision Transformer architecture
- Inspired by modern multi-label classification research
- Optimized for anime/artwork domain

---

This system represents a comprehensive solution for large-scale anime image tagging, combining research-grade techniques with production-ready deployment capabilities.
