---
license: gpl-3.0
datasets:
- p1atdev/danbooru-2024
language:
- en
pipeline_tag: image-classification
---

# Camie Tagger

An advanced deep learning model for automatically tagging anime/manga illustrations with relevant tags across multiple categories, achieving **58.1% micro F1 score** (31.5% macro F1 score using the balanced threshold preset) across 70,527 possible tags on a test set of 20,116 samples. Trained on a single 3060.

## 🚀 Updates (August 2025)

Camie-tagger-v2 is out! https://huggingface.co/Camais03/camie-tagger-v2

## 🔑 Key Highlights

- **Efficient Training**: Completed on just a single RTX 3060 GPU (12GB VRAM)
- **Fast Convergence**: Trained on 7,024,392 samples (3.52 epochs) in 1,756,098 batches
- **Comprehensive Coverage**: 70,527 tags across 7 categories (general, character, copyright, artist, meta, rating, year)
- **Innovative Architecture**: Two-stage prediction model with EfficientNetV2 backbone, Embedding layer and cross-attention for tag context
- **Model Size**: Initial model (214M parameters), Refined model (424M parameters)
- **User-Friendly Interface**: Easy-to-use application with customizable thresholds

*This project demonstrates that high-quality anime image tagging models can be trained on consumer hardware with the right optimization techniques.*

## ✨ Features

- **Multi-category tagging system**: Handles general tags, characters, copyright (series), artists, meta information, and content ratings
- **High performance**: 53.4% micro F1 score (31.5% macro F1) across 70,527 possible tags
- **Windows compatibility**: Initial-only mode works on Windows without Flash Attention
- **Streamlit web interface**: User-friendly UI for uploading and analyzing images and a tag collection game
- **Adjustable threshold profiles**: Micro, Macro, Balanced, Category-specific, High Precision, and High Recall profiles
- **Fine-grained control**: Per-category threshold adjustments for precision-recall tradeoffs
- **Safetensors and ONNX**: Original pickle files available in /models
- **EfficientNetV2-L Backbone**: Backbone performance greatly improved by the refining embedding layer

## 📊 Performance Notes

The performance seems a little underwhelming when looking at macro scores especially for general tags (check tables in performance section). However I've found that the model is still generally good at predicting these tags and is very good at character and copyright tags. It's also the case that there are just so many tags to predict.

The good news is that the model shows clear evidence of being undertrained, with consistent improvement across training epochs:

```
Training Progress (Micro vs Macro F1):
Epoch 2:   Micro-F1: 0.595    Macro-F1: 0.226  
Epoch 3:   Micro-F1: 0.606    Macro-F1: 0.268 (+4.2%)
Epoch 3.5: Micro-F1: 0.611    Macro-F1: 0.290 (+2.2% only 0.5 epochs)
```

This makes sense as 3.5 epochs really isn't alot of training time at all.

**Micro vs Macro F1 Explained**:
- **Micro-F1**: Calculates metrics globally by considering each tag instance prediction. This metric is dominated by common tags and categories with many examples.
- **Macro-F1**: Calculates metrics for each tag independently then averages them. This gives equal weight to rare tags and common tags.

The significant improvement in Macro-F1 (+4% per epoch) suggests that longer training would especially benefit rare tag recognition, while Micro-F1 improvements are slowing down as common tags are already well-learned.

### Future Training Plans

I plan to continue training the model to further improve performance, especially for rare tags. However, each epoch takes approximately 1.5-2 weeks of overnight training on my current hardware.

**If you'd like to support further training on the complete dataset or my future projects, consider supporting me here:(https://ko-fi.com/camais). Your support will directly enable longer training runs and better models!**

After this project, I plan to move onto LLMs as I have lots of ideas on how to improve upon them. I will update this model based on community attention.

## 📈 Performance Analysis

### Overall Performance

#### INITIAL PREDICTIONS

| CATEGORY | PROFILE | THRESHOLD | MICRO-F1 | MACRO-F1 |
|----------|---------|-----------|----------|----------|
| overall | MICRO OPT | 0.326 | 0.611 | 0.290 |
| | MACRO OPT | 0.201 | 0.534 | 0.331 |
| | BALANCED | 0.258 | 0.581 | 0.315 |
| | HIGH PRECISION | 0.500 | 0.497 | 0.163 |
| | HIGH RECALL | 0.120 | 0.308 | 0.260 |
| artist | MICRO OPT | 0.262 | 0.474 | 0.295 |
| | MACRO OPT | 0.140 | 0.262 | 0.287 |
| | BALANCED | 0.258 | 0.474 | 0.298 |
| | HIGH PRECISION | 0.464 | 0.310 | 0.135 |
| | HIGH RECALL | 0.153 | 0.302 | 0.301 |
| character | MICRO OPT | 0.294 | 0.749 | 0.444 |
| | MACRO OPT | 0.161 | 0.608 | 0.517 |
| | BALANCED | 0.258 | 0.746 | 0.478 |
| | HIGH PRECISION | 0.500 | 0.655 | 0.268 |
| | HIGH RECALL | 0.100 | 0.336 | 0.386 |
| copyright | MICRO OPT | 0.334 | 0.789 | 0.325 |
| | MACRO OPT | 0.205 | 0.700 | 0.404 |
| | BALANCED | 0.258 | 0.763 | 0.377 |
| | HIGH PRECISION | 0.500 | 0.747 | 0.209 |
| | HIGH RECALL | 0.100 | 0.347 | 0.267 |
| general | MICRO OPT | 0.322 | 0.607 | 0.180 |
| | MACRO OPT | 0.225 | 0.537 | 0.210 |
| | BALANCED | 0.258 | 0.576 | 0.204 |
| | HIGH PRECISION | 0.500 | 0.482 | 0.095 |
| | HIGH RECALL | 0.124 | 0.301 | 0.161 |
| meta | MICRO OPT | 0.330 | 0.601 | 0.134 |
| | MACRO OPT | 0.209 | 0.487 | 0.143 |
| | BALANCED | 0.258 | 0.557 | 0.144 |
| | HIGH PRECISION | 0.500 | 0.458 | 0.081 |
| | HIGH RECALL | 0.120 | 0.309 | 0.103 |
| rating | MICRO OPT | 0.359 | 0.808 | 0.791 |
| | MACRO OPT | 0.359 | 0.808 | 0.791 |
| | BALANCED | 0.258 | 0.779 | 0.768 |
| | HIGH PRECISION | 0.500 | 0.738 | 0.686 |
| | HIGH RECALL | 0.100 | 0.650 | 0.611 |
| year | MICRO OPT | 0.266 | 0.332 | 0.285 |
| | MACRO OPT | 0.258 | 0.331 | 0.286 |
| | BALANCED | 0.258 | 0.331 | 0.286 |
| | HIGH PRECISION | 0.302 | 0.308 | 0.251 |
| | HIGH RECALL | 0.213 | 0.304 | 0.279 |

#### REFINED PREDICTIONS

| CATEGORY | PROFILE | THRESHOLD | MICRO-F1 | MACRO-F1 |
|----------|---------|-----------|----------|----------|
| overall | MICRO OPT | 0.326 | 0.613 | 0.295 |
| | MACRO OPT | 0.193 | 0.546 | 0.338 |
| | BALANCED | 0.262 | 0.586 | 0.326 |
| | HIGH PRECISION | 0.500 | 0.499 | 0.173 |
| | HIGH RECALL | 0.120 | 0.310 | 0.262 |
| artist | MICRO OPT | 0.278 | 0.480 | 0.297 |
| | MACRO OPT | 0.148 | 0.288 | 0.299 |
| | BALANCED | 0.262 | 0.483 | 0.311 |
| | HIGH PRECISION | 0.480 | 0.314 | 0.140 |
| | HIGH RECALL | 0.153 | 0.302 | 0.306 |
| character | MICRO OPT | 0.302 | 0.757 | 0.460 |
| | MACRO OPT | 0.157 | 0.591 | 0.524 |
| | BALANCED | 0.262 | 0.751 | 0.496 |
| | HIGH PRECISION | 0.500 | 0.669 | 0.286 |
| | HIGH RECALL | 0.100 | 0.331 | 0.386 |
| copyright | MICRO OPT | 0.367 | 0.792 | 0.317 |
| | MACRO OPT | 0.189 | 0.671 | 0.419 |
| | BALANCED | 0.262 | 0.767 | 0.392 |
| | HIGH PRECISION | 0.492 | 0.755 | 0.228 |
| | HIGH RECALL | 0.100 | 0.349 | 0.270 |
| general | MICRO OPT | 0.326 | 0.608 | 0.181 |
| | MACRO OPT | 0.237 | 0.553 | 0.215 |
| | BALANCED | 0.262 | 0.580 | 0.208 |
| | HIGH PRECISION | 0.500 | 0.484 | 0.100 |
| | HIGH RECALL | 0.124 | 0.303 | 0.165 |
| meta | MICRO OPT | 0.330 | 0.602 | 0.127 |
| | MACRO OPT | 0.197 | 0.468 | 0.145 |
| | BALANCED | 0.262 | 0.563 | 0.152 |
| | HIGH PRECISION | 0.500 | 0.453 | 0.087 |
| | HIGH RECALL | 0.120 | 0.305 | 0.107 |
| rating | MICRO OPT | 0.375 | 0.808 | 0.787 |
| | MACRO OPT | 0.338 | 0.809 | 0.795 |
| | BALANCED | 0.262 | 0.784 | 0.773 |
| | HIGH PRECISION | 0.500 | 0.735 | 0.678 |
| | HIGH RECALL | 0.100 | 0.652 | 0.610 |
| year | MICRO OPT | 0.266 | 0.332 | 0.292 |
| | MACRO OPT | 0.258 | 0.331 | 0.293 |
| | BALANCED | 0.262 | 0.333 | 0.293 |
| | HIGH PRECISION | 0.306 | 0.306 | 0.255 |
| | HIGH RECALL | 0.209 | 0.301 | 0.275 |

The model performs particularly well on character identification (75.7% F1 across 26,968 tags), copyright/series detection (79.2% F1 across 5,364 tags), and content rating classification (80.8% F1 across 4 tags).

### Initial vs. Refined Prediction Performance

| PREDICTION TYPE | MICRO-F1 | MACRO-F1 | PRECISION | RECALL |
|-----------------|----------|----------|-----------|---------|
| INITIAL | 0.611 | 0.290 | 0.610 | 0.613 |
| REFINED | 0.613 | 0.295 | 0.617 | 0.609 |

Refinement improves Micro-F1 by +0.2% and Macro-F1 by +0.5%. As shown, the refined predictions offer a small but consistent improvement over the initial predictions, making the Initial-only model a good choice for Windows users where Flash Attention isn't available.

### Real-world Tag Accuracy

In personal testing, I've observed that many "false positives" according to the benchmark are actually correct tags that were missing from the Danbooru dataset (which itself is not 100% perfectly tagged). Some observations:
- For character, copyright, and artist categories, the top predicted tag is frequently correct even when the model isn't highly confident.
- Many seemingly incorrect general tags are actually appropriate descriptors that were simply not included in the original tagging.

For these reasons, the **High Recall** threshold profile may produce better perceived results in practice despite a lower formal F1 score. When using the application, limiting the output to the top N tags per category can deliver the most accurate and useful results.

### Comparison to WDTaggers:
It's a little tricky to compare our taggers for two reasons:
  The amount of tags 70,000+ for mine and 10,000+ for WD.
  I only kept samples with at least 25 general tags (and at least 1 character+copyright tag) vs at least 10 for WD. This means my samples on average have 44.74 mean tags per sample against WD taggers 36.48 in my own testing. That is a difference of 18.5%.

Keeping that in mind for the current checkpoint I'd say that mine seems somewhat more accurate for rarer tags just because mine covers much more. WD is considerably more accurate for common tags.

Here is a comparison between the two taggers with the same number of tags: https://huggingface.co/datasets/SmilingWolf/camie-tagger-vs-wd-tagger-val

In my personal testing mine seems to be better at rarer characters, copyright, artists and some rare general tags getting alternative costumes and the artist etc. Camie tagger did seem to have a few 1-3 more false positives for the general tags however. Keep in mind this is with a couple of images so I could be wrong.

After a few more epochs I think the gap could be a lot smaller but it's going to be a month or so before that point while I let it train overnight. I think WD tagger was trained for 50+ epochs with mine currently at 3.5.

Overall the distribution of tags is extremely long tailed. My game shows this with the rarity range. Most tags end up in the most rare category. Both taggers should give you good accuracy for the most common tags. 

## 🛠️ Requirements

- **Python 3.11.9 specifically** (newer versions are incompatible)
- PyTorch 1.10+
- Streamlit
- PIL/Pillow
- NumPy
- Flash Attention (note: doesn't work properly on Windows only needed for refined model which I'm not supporting that much anyway)

## 🔧 Usage

Setup the application and game by executing `setup.bat`. This installs the required virtual environment:

- Upload your own images or select from example images
- Choose different threshold profiles
- Adjust category-specific thresholds
- View predictions organized by category
- Filter and sort tags based on confidence

Use run_app.bat and run_game.bat.

## 🎮 Tag Collector Game (Camie Collector)

Introducing a Tagging game - a gamified approach to anime image tagging that helps you understand the performance and limits of the model. This was a shower thought gone to far! Lots of Project Moon references.

### How to Play:
1. Upload an image
2. Scan for tags to discover them
   ![Collect Tags Tab](images/collect_tags.PNG)
3. Earn TagCoins for new discoveries
4. Spend TagCoins on upgrades to lower the threshold
   ![Upgrades Tab](images/upgrades.PNG)
5. Lower thresholds reveal rarer tags!
6. Collect sets of related tags for bonuses and reveal unique mosaics!
   ![Mosaics Tab](images/mosaics.PNG)
7. Visit the Library System to discover unique tags (not collect)
   ![Library Tab](images/library.PNG)
8. Use collected tags to either inspire new searches or generate essence
9. Use Enkephalin to generate Tag Essences
   ![Essence Tab](images/essence_tab.PNG)
10. Use the Tag Essence Generator to collect the tag and related tags to it. Lamp Essence:
    ![Lamp Essence](images/lamp_essence.jpg)

## 🖥️ Web Interface Guide

The interface is divided into three main sections:

1. **Model Selection** (Sidebar):
   - Choose between Full Model, Initial-only Model or ONNX accelerated (initial only)
   - View model information and memory usage

2. **Image Upload** (Left Panel):
   - Upload your own images or select from examples
   - View the selected image

3. **Tagging Controls** (Right Panel):
   - Select threshold profile
   - Adjust thresholds for precision-recall and micro/macro tradeoff
   - Configure display options
   - View predictions organized by category

### Display Options:

- **Show all tags**: Display all tags including those below threshold
- **Compact view**: Hide progress bars for cleaner display
- **Minimum confidence**: Filter out low-confidence predictions
- **Category selection**: Choose which categories to include in the summary

### Interface Screenshots:

![Application Interface](images/app_screenshot.png)

![Tag Results Example](images/tag_results_example.png)

## 🧠 Training Details

### Dataset

The model was trained on a carefully filtered subset of the [Danbooru 2024 dataset](https://huggingface.co/datasets/p1atdev/danbooru-2024), which contains a vast collection of anime/manga illustrations with comprehensive tagging.

#### Filtering Process:

The dataset was filtered with the following constraints:

```python
# Minimum tags per category required for each image
min_tag_counts = {
    'general': 25, 
    'character': 1, 
    'copyright': 1, 
    'artist': 0, 
    'meta': 0
}

# Minimum samples per tag required for tag to be included
min_tag_samples = {
    'general': 20, 
    'character': 40, 
    'copyright': 50, 
    'artist': 200, 
    'meta': 50
}
```

This filtering process:
1. First removed low-sample tags (tags with fewer occurrences than specified in `min_tag_samples`)
2. Then removed images with insufficient tags per category (as specified in `min_tag_counts`)

#### Training Data:

- **Starting dataset size**: ~3,000,000 filtered images
- **Training subset**: 2,000,000 images (due to storage and time constraints)
- **Training duration**: 3.5 epochs

#### Preprocessing:

Images were preprocessed with minimal transformations:
- Tensor normalization (scaled to 0-1 range)
- Resized while maintaining original aspect ratio
- No additional augmentations were applied

### Loss Function

The model employs a specialized `UnifiedFocalLoss` to address the extreme class imbalance inherent in multi-label tag prediction:

```python
class UnifiedFocalLoss(nn.Module):
    def __init__(self, device=None, gamma=2.0, alpha=0.25, lambda_initial=0.4):
        # Implementation details...
```

#### Key Components:

1. **Focal Loss Mechanism**:
   - Down-weights well-classified examples (γ=2.0) to focus training on difficult tags
   - Addresses the extreme imbalance between positive and negative examples (often 100:1 or worse)
   - Uses α=0.25 to balance positive/negative examples across 70,527 possible tags

2. **Two-stage Weighting**:
   - Combines losses from both prediction stages (`initial_predictions` and `refined_predictions`)
   - Uses λ=0.4 to weight the initial prediction loss, giving more importance (0.6) to refined predictions
   - This encourages the model to improve predictions in the refinement stage while still maintaining strong initial predictions

3. **Per-sample Statistics**:
   - Tracks separate metrics for positive and negative samples
   - Provides detailed debugging information about prediction distributions
   - Enables analysis of which tag categories are performing well/poorly

This loss function was essential for achieving high F1 scores across diverse tag categories despite the extreme classification challenge of 70,527 possible tags.

### DeepSpeed Configuration

Microsoft DeepSpeed was crucial for training this model on consumer hardware. The project uses a carefully tuned configuration to maximize efficiency:

```python
def create_deepspeed_config(
    config_path,
    learning_rate=3e-4,
    weight_decay=0.01,
    num_train_samples=None,
    micro_batch_size=4,
    grad_accum_steps=8
):
    # Implementation details...
```

#### Key Optimizations:

1. **Memory Efficiency**:
   - **ZeRO Stage 2**: Partitions optimizer states and gradients, dramatically reducing memory requirements
   - **Activation Checkpointing**: Trades computation for memory by recomputing activations during backpropagation
   - **Contiguous Memory Optimization**: Reduces memory fragmentation

2. **Mixed Precision Training**:
   - **FP16 Mode**: Uses half-precision (16-bit) for most calculations, with automatic loss scaling
   - **Initial Scale Power**: Set to 16 for stable convergence with large batch sizes

3. **Gradient Accumulation**:
   - Micro-batch size of 4 with 8 gradient accumulation steps
   - Effective batch size of 32 while only requiring memory for 4 samples at once

4. **Learning Rate Schedule**:
   - WarmupLR scheduler with gradual increase from 3e-6 to 3e-4
   - Warmup over 1/4 of an epoch to stabilize early training

This configuration allowed the model to train efficiently with only 12GB of VRAM while maintaining numerical stability across millions of training examples with 70,527 output dimensions.

### Model Architecture

The model uses a novel two-stage prediction approach that achieves superior performance compared to traditional single-stage models:

#### Image Feature Extraction:
- **Backbone**: EfficientNet V2-L extracts high-quality visual features from input images
- **Spatial Pooling**: Adaptive averaging converts spatial features to a compact 1280-dimensional embedding

#### Initial Prediction Stage:
- Direct classification from image features through a multi-layer classifier
- Bottleneck architecture with LayerNorm and GELU activations between linear layers
- Outputs initial tag probabilities across all 70,527 possible tags
- Model size: 214,657,273 parameters

#### Tag Context Mechanism:
- Top predicted tags are embedded using a shared embedding space
- Self-attention layer allows tags to influence each other based on co-occurrence patterns
- Normalized tag embeddings represent a coherent "tag context" for the image

#### Cross-Attention Refinement:
- Image features and tag embeddings interact through cross-attention
- Each dimension of the image features attends to relevant dimensions in the tag space
- This creates a bidirectional flow of information between visual features and semantic tags

#### Refined Predictions:
- Fused features (original + cross-attended) feed into a final classifier
- Residual connection ensures initial predictions are preserved when beneficial
- Temperature scaling provides calibrated probability outputs
- Total model size: 424,793,720 parameters

This dual-stage approach allows the model to leverage tag co-occurrence patterns and semantic relationships, improving accuracy without increasing the parameter count significantly.

### Model Details

#### Tag Categories:

The model recognizes tags across these categories:
- **General**: Visual elements, concepts, clothing, etc. (30,841 tags)
- **Character**: Individual characters appearing in the image (26,968 tags)
- **Copyright**: Source material (anime, manga, game) (5,364 tags)
- **Artist**: Creator of the artwork (7,007 tags)
- **Meta**: Meta information about the image (323 tags)
- **Rating**: Content rating (4 tags)
- **Year**: Year of upload (20 tags)

All supported tags are stored in `model/metadata.json`, which maps tag IDs to their names and categories.

## 💻 Training Environment

The model was trained using surprisingly modest hardware:

- **GPU**: Single NVIDIA RTX 3060 (12GB VRAM)
- **RAM**: 64GB system memory
- **Platform**: Windows with WSL (Windows Subsystem for Linux)
- **Libraries**: 
  - Microsoft DeepSpeed for memory-efficient training
  - PyTorch with CUDA acceleration
  - Flash Attention for optimized attention computation

### Training Notebooks

The repository includes two main training notebooks:

1. **CAMIE Tagger.ipynb**:
   - Main training notebook
   - Dataset loading and preprocessing
   - Model initialization
   - Initial training loop with DeepSpeed integration
   - Tag selection optimization
   - Metric tracking and visualization

2. **Camie Tagger Cont and Evals.ipynb**:
   - Continuation of training from checkpoints
   - Comprehensive model evaluation
   - Per-category performance metrics
   - Threshold optimization
   - Model conversion for deployment in the app
   - Export functionality for the standalone application

### Training Monitor

The project includes a real-time training monitor accessible via browser at `localhost:5000` during training:

#### Performance Tips:

⚠️ **Important**: For optimal training speed, keep VSCode minimized and the training monitor open in your browser. This can improve iteration speed by **3-5x** due to how the Windows/WSL graphics stack handles window focus and CUDA kernel execution.

#### Monitor Features:

The training monitor provides three main views:

##### 1. Overview Tab:

![Overview Tab](images/training_monitor_overview.png)

- **Training Progress**: Real-time metrics including epoch, batch, speed, and time estimates
- **Loss Chart**: Training and validation loss visualization
- **F1 Scores**: Initial and refined F1 metrics for both training and validation

##### 2. Predictions Tab:

![Predictions Tab](images/training_monitor_predictions.png)

- **Image Preview**: Shows the current sample being analyzed
- **Prediction Controls**: Toggle between initial and refined predictions
- **Tag Analysis**: 
  - Color-coded tag results (correct, incorrect, missing)
  - Confidence visualization with probability bars
  - Category-based organization
  - Filtering options for error analysis

##### 3. Selection Analysis Tab:

![Selection Analysis Tab](images/training_monitor_selection.png)

- **Selection Metrics**: Statistics on tag selection quality
  - Ground truth recall
  - Average probability for ground truth vs. non-ground truth tags
  - Unique tags selected
- **Selection Graph**: Trends in selection quality over time
- **Selected Tags Details**: Detailed view of model-selected tags with confidence scores

The monitor provides invaluable insights into how the two-stage prediction model is performing, particularly how the tag selection process is working between the initial and refined prediction stages.

### Training Notes:

- Training notebooks require WSL and likely 32GB+ of RAM to handle the dataset
- Microsoft DeepSpeed was crucial for fitting the model and batches into the available VRAM
- Despite hardware limitations, the model achieves impressive results
- With more computational resources, the model could be trained longer on the full dataset

## 🙏 Acknowledgments

- Claude Sonnet 3.5 and 3.7 for being incredibly helpful with the brainstorming and coding
- [EfficientNetV2](https://arxiv.org/abs/2104.00298) for the backbone
- [Danbooru](https://danbooru.donmai.us/) for the incredible dataset of tagged anime images
- [p1atdev](https://huggingface.co/p1atdev) for the processed Danbooru 2024 dataset
- Microsoft for DeepSpeed, which made training possible on consumer hardware
- PyTorch and the open-source ML community