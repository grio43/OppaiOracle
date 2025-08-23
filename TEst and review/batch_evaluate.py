#!/usr/bin/env python3
import json, os, sys, time, pathlib, contextlib
from collections import defaultdict
from typing import List, Tuple, Dict, Set
import numpy as np
from PIL import Image
import onnxruntime as ort
import torch
from tqdm import tqdm
import gc
import psutil  # For system memory monitoring

def get_system_memory_info():
    """Get current system memory usage"""
    mem = psutil.virtual_memory()
    return mem.used / 1024**3, mem.available / 1024**3  # GB

def get_gpu_memory_info():
    """Get GPU memory info using nvidia-ml-py if available, otherwise nvidia-smi"""
    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        used = info.used / 1024**3  # GB
        total = info.total / 1024**3  # GB
        return used, total
    except:
        # Fallback: parse nvidia-smi output
        import subprocess
        try:
            result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used,memory.total', 
                                   '--format=csv,nounits,noheader'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                used, total = map(float, result.stdout.strip().split(','))
                return used/1024, total/1024  # Convert MB to GB
        except:
            pass
        return 0, 0

def load_tag_names(config_path):
    if config_path and os.path.exists(config_path):
        with open(config_path, "r") as f:
            cfg = json.load(f)
        names = cfg.get("tag_names", None)
        norm = cfg.get("normalization_params", None)
        return names, norm
    return None, None

def letterbox_resize_numpy(image, target_size=640, pad_color=(114, 114, 114), patch_size=16):
    """
    Letterbox resize for numpy arrays with padding info tracking.
    Matches the training letterbox_resize but works with PIL/numpy.
    """
    h, w = image.shape[:2] if len(image.shape) == 3 else image.shape
    
    # Compute scale to fit image in target_size
    scale = min(target_size / h, target_size / w)
    new_h = int(round(h * scale))
    new_w = int(round(w * scale))
    
    # Resize image
    from PIL import Image as PILImage
    if isinstance(image, np.ndarray):
        pil_img = PILImage.fromarray(image)
    else:
        pil_img = image
    resized = pil_img.resize((new_w, new_h), PILImage.Resampling.LANCZOS)
    resized_arr = np.array(resized)
    
    # Calculate padding
    pad_h = target_size - new_h
    pad_w = target_size - new_w
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left
    
    # Ensure divisibility by patch_size
    final_h = target_size
    final_w = target_size
    extra_pad_bottom = 0
    extra_pad_right = 0
    
    if patch_size > 1:
        if final_h % patch_size != 0:
            extra_pad_bottom = patch_size - (final_h % patch_size)
        if final_w % patch_size != 0:
            extra_pad_right = patch_size - (final_w % patch_size)
    
    final_h += extra_pad_bottom
    final_w += extra_pad_right
    
    # Create padded canvas
    if len(resized_arr.shape) == 3:
        canvas = np.full((final_h, final_w, 3), pad_color, dtype=np.uint8)
        canvas[pad_top:pad_top + new_h, pad_left:pad_left + new_w] = resized_arr
    else:
        canvas = np.full((final_h, final_w), pad_color[0], dtype=np.uint8)
        canvas[pad_top:pad_top + new_h, pad_left:pad_left + new_w] = resized_arr
    
    padding_info = {
        'scale': scale,
        'pad': (pad_left, pad_top, pad_right + extra_pad_right, pad_bottom + extra_pad_bottom),
        'out_size': (final_h, final_w),
        'in_size': (h, w)
    }
    
    return canvas, padding_info

def build_padding_mask_from_info(batch_infos, H, W, device='cpu'):
    """Build pixel-level padding masks from letterbox info."""
    batch_size = len(batch_infos)
    pmasks = torch.ones((batch_size, 1, H, W), dtype=torch.float32, device=device)
    
    for i, info in enumerate(batch_infos):
        l, t, r, b = info['pad']
        if t > 0:
            pmasks[i, :, :t, :] = 0
        if b > 0:
            pmasks[i, :, H-b:, :] = 0
        if l > 0:
            pmasks[i, :, :, :l] = 0
        if r > 0:
            pmasks[i, :, :, W-r:] = 0
    
    return pmasks

def downsample_mask_to_patches(pixel_mask, patch_size=16, threshold=0.9):
    """Downsample pixel mask to patch-level mask for ViT."""
    # Average pool the mask
    pooled = torch.nn.functional.avg_pool2d(pixel_mask, kernel_size=patch_size, stride=patch_size)
    # Threshold to get binary mask (True = valid content)
    token_mask = pooled >= threshold
    return token_mask

def preprocess_batch(image_paths, image_size=640, mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5),
                    gray_background=(114, 114, 114)):
    """Preprocess multiple images into a batch with proper transparency handling and padding"""
    batch = []
    valid_indices = []
    batch_infos = []

    for idx, image_path in enumerate(image_paths):
        try:
            img = Image.open(image_path)

            # Handle transparency by compositing on gray background
            if img.mode in ('RGBA', 'LA') or (img.mode == 'P' and 'transparency' in img.info):
                # Create gray background
                background = Image.new('RGB', img.size, gray_background)
                # Convert image to RGBA if not already
                if img.mode != 'RGBA':
                    img = img.convert('RGBA')
                # Composite image over gray background
                background.paste(img, mask=img.split()[3])  # Use alpha channel as mask
                img = background
            else:
                img = img.convert('RGB')

            # Use letterbox resize instead of thumbnail + center padding
            img_array = np.array(img)
            letterboxed, pad_info = letterbox_resize_numpy(img_array, image_size, gray_background)

            # Convert to float and normalize
            arr = letterboxed.astype(np.float32) / 255.0
            # HWC -> CHW
            arr = arr.transpose(2, 0, 1)
            # Normalize
            mean_arr = np.array(mean, dtype=np.float32)[:, None, None]
            std_arr = np.array(std, dtype=np.float32)[:, None, None]
            arr = (arr - mean_arr) / std_arr
            batch.append(arr)
            batch_infos.append(pad_info)
            valid_indices.append(idx)
        except Exception as e:
            print(f"Error preprocessing {image_path}: {e}")
            continue

    if batch:
        return np.stack(batch, axis=0).astype(np.float32), valid_indices, batch_infos
    return None, [], []

def get_predictions_from_scores(scores, tag_names, threshold=0.5):
    """Get predicted tags above threshold for batch or single image"""
    if scores.ndim == 1:
        scores = scores[np.newaxis, :]  # Add batch dimension if missing
    
    batch_predictions = []
    for score_vec in scores:
        predicted_tags = set()
        for i, score in enumerate(score_vec):
            if score >= threshold:
                tag = tag_names[i] if tag_names and i < len(tag_names) else f"tag_{i}"
                predicted_tags.add(tag)
        batch_predictions.append(predicted_tags)
    
    return batch_predictions

def parse_ground_truth_tags(json_data):
    """Parse ground truth tags from JSON"""
    tags_str = json_data.get("tags", "")
    tags = set(tag.strip() for tag in tags_str.split() if tag.strip())
    
    rating = json_data.get("rating", "")
    if rating and rating != "safe":
        tags.add(rating)
    
    return tags

def calculate_metrics(predicted: Set[str], ground_truth: Set[str]) -> Dict[str, float]:
    """Calculate precision, recall, and F1 score"""
    if not predicted and not ground_truth:
        return {
            "precision": 1.0,
            "recall": 1.0,
            "f1": 1.0,
            "true_positives": 0,
            "false_positives": 0,
            "false_negatives": 0,
        }

    if not predicted:
        return {
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "true_positives": 0,
            "false_positives": 0,
            "false_negatives": len(ground_truth),
        }

    if not ground_truth:
        return {
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "true_positives": 0,
            "false_positives": len(predicted),
            "false_negatives": 0,
        }

    true_positives = len(predicted & ground_truth)
    false_positives = len(predicted - ground_truth)
    false_negatives = len(ground_truth - predicted)

    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) else 0.0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
    }

def create_gpu_session(model_path, max_memory_gb=25):
    """Create ONNX Runtime session with GPU configuration"""
    providers = []

    # Configure CUDA provider with memory limits
    cuda_options = {
        'device_id': 0,
        'arena_extend_strategy': 'kSameAsRequested',
        'gpu_mem_limit': int(max_memory_gb * 1024 * 1024 * 1024),
        'cudnn_conv_algo_search': 'HEURISTIC',
        'do_copy_in_default_stream': True,
        'cudnn_conv_use_max_workspace': True,
        'enable_cuda_graph': False,  # Disable CUDA graphs to reduce memory fragmentation
    }
    
    # Try CUDA first, then TensorRT if available
    if 'CUDAExecutionProvider' in ort.get_available_providers():
        providers.append(('CUDAExecutionProvider', cuda_options))
        print(f"Using CUDA provider with {max_memory_gb}GB memory limit")
    
    if 'TensorrtExecutionProvider' in ort.get_available_providers():
        trt_options = {
            'device_id': 0,
            'trt_max_workspace_size': int(8 * 1024 * 1024 * 1024),  # 8GB for TRT workspace
            'trt_fp16_enable': True,  # Enable FP16 for better performance
        }
        providers.append(('TensorrtExecutionProvider', trt_options))
        print("TensorRT provider available and configured")
    
    # Fallback to CPU if no GPU providers available
    providers.append('CPUExecutionProvider')
    
    # Create session options for better performance
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL  # Use sequential for more predictable memory
    sess_options.enable_mem_pattern = False  # Disable memory pattern optimization to reduce RAM usage
    sess_options.inter_op_num_threads = 4
    sess_options.intra_op_num_threads = 4
    
    # Create session
    session = ort.InferenceSession(model_path, sess_options=sess_options, providers=providers)
    
    # Print which provider is actually being used
    actual_provider = session.get_providers()[0]
    print(f"Session created with provider: {actual_provider}")

    # Warmup run to initialize GPU memory pools
    try:
        dummy_input = np.random.randn(1, 3, 640, 640).astype(np.float32)
        _ = session.run(None, {session.get_inputs()[0].name: dummy_input})
        print("Warmup run completed")
    except Exception as e:
        print(f"Warmup failed (non-critical): {e}")

    return session

def create_pytorch_session(model_path, device='cuda', compile_model=False):
    """Create PyTorch session with optimizations."""
    import torch

    # Load model
    if model_path.endswith('.onnx'):
        # Convert ONNX to PyTorch if needed
        print("Note: For best PyTorch performance, use a native .pt checkpoint")
        return None

    # Assume it's a PyTorch checkpoint
    checkpoint = torch.load(model_path, map_location=device)

    # Extract model from checkpoint
    if 'model_state_dict' in checkpoint:
        # Need to reconstruct model architecture
        from model_architecture import create_model
        model = create_model(
            num_tags=checkpoint.get('num_tags', 10000),
            num_ratings=checkpoint.get('num_ratings', 5)
        )
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model = checkpoint

    model.to(device)
    model.eval()

    # Enable optimizations
    if device == 'cuda':
        # Use channels_last memory format
        model = model.to(memory_format=torch.channels_last)

        # Enable TF32 for Ampere+ GPUs
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        # Enable cuDNN benchmark for consistent input sizes
        torch.backends.cudnn.benchmark = True

        # Compile model for additional speedup (PyTorch 2.0+)
        if compile_model and hasattr(torch, 'compile'):
            print("Compiling model with torch.compile()...")
            model = torch.compile(model)

    return model

def process_batch_pytorch(data_folder, model_path, config_path=None, threshold=0.5,
                         image_size=640, output_file="evaluation_results_torch.json",
                         batch_size=32, limit=None, compile_model=False):
    """Process images using PyTorch with full GPU optimizations."""

    # Check CUDA availability
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name()}")
        # Set memory fraction to prevent OOM
        torch.cuda.set_per_process_memory_fraction(0.8)

    # Load configuration
    tag_names, norm = load_tag_names(config_path)
    mean, std = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
    if norm and "mean" in norm and "std" in norm:
        mean, std = tuple(norm["mean"]), tuple(norm["std"])

    # Create PyTorch model
    model = create_pytorch_session(model_path, device, compile_model)
    if model is None:
        print("Failed to load PyTorch model, falling back to ONNX")
        return None

    # Find all JSON files
    json_files = sorted(pathlib.Path(data_folder).glob("*.json"))
    if limit:
        json_files = json_files[:limit]

    print(f"Found {len(json_files)} JSON files to process")
    print(f"Batch size: {batch_size}")

    # Create DataLoader for efficient batching
    from torch.utils.data import DataLoader, Dataset

    class EvalDataset(Dataset):
        def __init__(self, json_files):
            self.samples = []
            for json_path in json_files:
                try:
                    with open(json_path, 'r') as f:
                        json_data = json.load(f)
                    image_filename = json_data.get("filename")
                    if image_filename:
                        image_path = json_path.parent / image_filename
                        if image_path.exists():
                            self.samples.append({
                                'image_path': str(image_path),
                                'ground_truth': parse_ground_truth_tags(json_data),
                                'filename': image_filename
                            })
                except Exception as e:
                    print(f"Error loading {json_path}: {e}")

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, idx):
            return self.samples[idx]

    dataset = EvalDataset(json_files)

    def _collate_keep_list(batch):
        return batch

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True,
        collate_fn=_collate_keep_list,
    )

    all_results = []
    overall_metrics = defaultdict(list)
    tag_performance = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    total_inference_time = 0.0

    use_amp = device.type == 'cuda'
    amp_dtype = torch.bfloat16 if (use_amp and torch.cuda.is_bf16_supported()) else torch.float16

    with torch.no_grad():
        autocast_ctx = torch.cuda.amp.autocast(dtype=amp_dtype) if use_amp else contextlib.nullcontext()
        with autocast_ctx:
            for batch in tqdm(dataloader, desc="Processing batches"):
                image_paths = [s['image_path'] for s in batch]
                gt_sets = [s['ground_truth'] for s in batch]
                filenames = [s['filename'] for s in batch]

                batch_input, valid_idx, batch_infos = preprocess_batch(
                    image_paths, image_size, mean, std
                )
                if batch_input is None:
                    continue

                images = torch.from_numpy(batch_input).to(device, non_blocking=True)
                images = images.to(memory_format=torch.channels_last)

                H, W = images.shape[-2:]
                padding_mask = build_padding_mask_from_info(batch_infos, H, W, device)
                if torch.all(padding_mask == 1):
                    padding_mask = None

                start_time = time.time()
                outputs = model(images, padding_mask=padding_mask)
                if use_amp:
                    torch.cuda.synchronize()
                total_inference_time += time.time() - start_time

                if isinstance(outputs, dict) and 'tag_logits' in outputs:
                    logits = outputs['tag_logits']
                else:
                    logits = outputs
                scores = logits.sigmoid().float().cpu().numpy()

                preds = get_predictions_from_scores(scores, tag_names, threshold)

                for j, src_idx in enumerate(valid_idx):
                    pred = preds[j]
                    gt = gt_sets[src_idx]
                    metrics = calculate_metrics(pred, gt)
                    for t in (pred & gt):
                        tag_performance[t]["tp"] += 1
                    for t in (pred - gt):
                        tag_performance[t]["fp"] += 1
                    for t in (gt - pred):
                        tag_performance[t]["fn"] += 1
                    all_results.append({
                        "filename": filenames[src_idx],
                        "ground_truth_tags": sorted(gt),
                        "predicted_tags": sorted(pred),
                        "metrics": metrics,
                    })
                    for k, v in metrics.items():
                        if isinstance(v, (int, float)):
                            overall_metrics[k].append(v)

    summary = {
        "total_files": len(dataset),
        "processed_files": len(all_results),
        "failed_files": len(dataset) - len(all_results),
        "average_metrics": {},
    }
    for k, v in overall_metrics.items():
        if v:
            summary["average_metrics"][k] = float(np.mean(v))
            summary["average_metrics"][f"{k}_std"] = float(np.std(v))

    tag_f1_scores = []
    for tag, stats in tag_performance.items():
        tp, fp, fn = stats["tp"], stats["fp"], stats["fn"]
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        tag_f1_scores.append({
            "tag": tag,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "support": tp + fn,
        })
    tag_f1_scores.sort(key=lambda x: x["f1"], reverse=True)
    summary["tag_performance"] = {
        "best_tags": tag_f1_scores[:10],
        "worst_tags": tag_f1_scores[-10:] if len(tag_f1_scores) > 10 else [],
    }
    summary["total_inference_time"] = total_inference_time
    summary["images_per_second"] = len(dataset) / total_inference_time if total_inference_time else 0.0

    with open(output_file, "w") as f:
        json.dump({"summary": summary, "detailed_results": all_results}, f, indent=2)

    print(f"Total inference time: {total_inference_time:.2f}s")
    print(f"Images per second: {summary['images_per_second']:.2f}")

def process_batch_gpu(data_folder, model_path, config_path=None, threshold=0.5, 
                     image_size=640, output_file="evaluation_results.json",
                     batch_size=32, limit=None, max_memory_gb=25):
    """Process images in batches on GPU for efficient inference"""
    
    # Load model configuration
    tag_names, norm = load_tag_names(config_path)
    mean, std = (0.5,0.5,0.5), (0.5,0.5,0.5)
    if norm and "mean" in norm and "std" in norm:
        mean, std = tuple(norm["mean"]), tuple(norm["std"])
    
    # Initialize GPU session
    sess = create_gpu_session(model_path, max_memory_gb)
    
    # Get input/output names
    output_names = [o.name for o in sess.get_outputs()]
    use_scores = "scores" if "scores" in output_names else output_names[0]
    
    input_names = [i.name for i in sess.get_inputs()]
    input_name = input_names[0] if input_names else "input_image"
    
    # Print session info
    print(f"Model input: {input_name}, shape: {sess.get_inputs()[0].shape}")
    print(f"Model output: {use_scores}, shape: {sess.get_outputs()[0].shape}")
    
    # Find all JSON files
    json_files = list(pathlib.Path(data_folder).glob("*.json"))
    if limit:
        json_files = json_files[:limit]
    
    print(f"Found {len(json_files)} JSON files to process")
    print(f"Batch size: {batch_size}")
    
    # Results storage
    all_results = []
    overall_metrics = defaultdict(list)
    tag_performance = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    failed_files = []
    
    # Process in batches
    total_batches = (len(json_files) + batch_size - 1) // batch_size
    total_inference_time = 0

    # Monitor initial memory
    ram_used_init, ram_avail_init = get_system_memory_info()
    gpu_used_init, gpu_total_init = get_gpu_memory_info()
    print(f"Initial RAM: {ram_used_init:.2f}GB used, {ram_avail_init:.2f}GB available")
    print(f"Initial GPU: {gpu_used_init:.2f}GB used / {gpu_total_init:.2f}GB total")

    processed_count = 0

    for batch_idx in tqdm(range(0, len(json_files), batch_size),
                          desc=f"Processing batches",
                          total=total_batches):
        
        batch_files = json_files[batch_idx:batch_idx + batch_size]
        batch_data = []
        batch_image_paths = []
        batch_ground_truths = []
        
        # Load batch data
        for json_path in batch_files:
            try:
                with open(json_path, "r") as f:
                    json_data = json.load(f)
                
                image_filename = json_data.get("filename")
                if not image_filename:
                    failed_files.append(str(json_path))
                    continue
                
                image_path = json_path.parent / image_filename
                if not image_path.exists():
                    failed_files.append(str(json_path))
                    continue
                
                ground_truth = parse_ground_truth_tags(json_data)
                
                batch_data.append({
                    "json_path": json_path,
                    "image_filename": image_filename,
                    "json_data": json_data
                })
                batch_image_paths.append(str(image_path))
                batch_ground_truths.append(ground_truth)
                
            except Exception as e:
                print(f"Error loading {json_path}: {e}")
                failed_files.append(str(json_path))
                continue
        
        if not batch_image_paths:
            continue
        
        # Preprocess batch
        batch_input, valid_indices, batch_infos = preprocess_batch(
            batch_image_paths, image_size, mean, std, gray_background=(114, 114, 114)
        )
        if batch_input is None:
            continue

        # Run batch inference
        try:
            inputs = {input_name: batch_input}  # Already float32 from preprocessing

            # Monitor memory periodically
            if processed_count % 100 == 0:
                ram_used, ram_avail = get_system_memory_info()
                gpu_used, gpu_total = get_gpu_memory_info()
                print(f"\n[After {processed_count} images] RAM: {ram_used:.2f}GB used, GPU: {gpu_used:.2f}/{gpu_total:.2f}GB")

            start_time = time.time()
            outs = sess.run([use_scores], inputs)
            batch_inference_time = time.time() - start_time
            total_inference_time += batch_inference_time

            scores = outs[0]
            
            # Get predictions for the batch
            batch_predictions = get_predictions_from_scores(scores, tag_names, threshold)
            
            # Process results for valid indices
            for idx_in_valid, orig_idx in enumerate(valid_indices):
                predicted_tags = batch_predictions[idx_in_valid]
                ground_truth = batch_ground_truths[orig_idx]
                data = batch_data[orig_idx]
                
                # Calculate metrics
                metrics = calculate_metrics(predicted_tags, ground_truth)
                
                # Update tag-level performance
                for tag in predicted_tags & ground_truth:
                    tag_performance[tag]["tp"] += 1
                for tag in predicted_tags - ground_truth:
                    tag_performance[tag]["fp"] += 1
                for tag in ground_truth - predicted_tags:
                    tag_performance[tag]["fn"] += 1
                
                # Store results
                result = {
                    "filename": data["image_filename"],
                    "ground_truth_tags": list(ground_truth),
                    "predicted_tags": list(predicted_tags),
                    "metrics": metrics,
                    "batch_inference_time": batch_inference_time / len(valid_indices),  # Average per image
                    "num_gt_tags": len(ground_truth),
                    "num_pred_tags": len(predicted_tags)
                }
                all_results.append(result)
                
                # Update overall metrics
                processed_count += 1

                # Update overall metrics
                for key, value in metrics.items():
                    if isinstance(value, (int, float)):
                        overall_metrics[key].append(value)

            # Clear batch data to free memory
            del batch_input, scores, outs

            # Periodic garbage collection and result saving
            if len(all_results) % 500 == 0:
                # Save intermediate results
                with open(output_file.replace('.json', f'_partial_{len(all_results)}.json'), 'w') as f:
                    json.dump({"partial_results": all_results[-500:]}, f)
                gc.collect()  # Force garbage collection
                        
        except Exception as e:
            print(f"Error during batch inference: {e}")
            for path in batch_image_paths:
                failed_files.append(path)
            continue
    
    # Calculate aggregate statistics
    summary = {
        "total_files": len(json_files),
        "processed_files": len(all_results),
        "failed_files": len(failed_files),
        "threshold": threshold,
        "batch_size": batch_size,
        "total_inference_time": total_inference_time,
        "average_batch_time": total_inference_time / total_batches if total_batches > 0 else 0,
        "images_per_second": len(all_results) / total_inference_time if total_inference_time > 0 else 0,
        "average_metrics": {},
        "tag_performance": {},
        "failed_file_list": failed_files
    }
    
    # Calculate averages
    for metric, values in overall_metrics.items():
        if values:
            summary["average_metrics"][metric] = np.mean(values)
            summary["average_metrics"][f"{metric}_std"] = np.std(values)
    
    # Calculate per-tag F1 scores
    tag_f1_scores = []
    for tag, counts in tag_performance.items():
        tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        tag_f1_scores.append({
            "tag": tag,
            "f1": f1,
            "precision": precision,
            "recall": recall,
            "support": tp + fn
        })
    
    # Sort tags by F1 score
    tag_f1_scores.sort(key=lambda x: x["f1"], reverse=True)
    summary["tag_performance"] = {
        "best_tags": tag_f1_scores[:10],
        "worst_tags": tag_f1_scores[-10:] if len(tag_f1_scores) > 10 else []
    }
    
    # Save detailed results
    with open(output_file, "w") as f:
        json.dump({
            "summary": summary,
            "detailed_results": all_results
        }, f, indent=2)
    
    # Print summary
    print("\n" + "="*50)
    print("EVALUATION SUMMARY")
    print("="*50)
    print(f"Processed: {summary['processed_files']}/{summary['total_files']} files")
    print(f"Failed: {summary['failed_files']} files")
    print(f"Batch size: {batch_size}")
    print(f"Threshold: {threshold}")
    print(f"Total inference time: {total_inference_time:.2f}s")
    print(f"Images per second: {summary['images_per_second']:.2f}")
    
    # Final memory check
    ram_used_final, ram_avail_final = get_system_memory_info()
    gpu_used_final, gpu_total_final = get_gpu_memory_info()
    print(f"\nFinal RAM: {ram_used_final:.2f}GB used (Δ{ram_used_final-ram_used_init:+.2f}GB)")
    print(f"Final GPU: {gpu_used_final:.2f}/{gpu_total_final:.2f}GB used (Δ{gpu_used_final-gpu_used_init:+.2f}GB)")
    
    print("\nAverage Metrics:")
    for metric, value in summary["average_metrics"].items():
        if not metric.endswith("_std"):
            std_key = f"{metric}_std"
            std_val = summary["average_metrics"].get(std_key, 0)
            print(f"  {metric}: {value:.4f} (±{std_val:.4f})")
    
    print("\nTop 5 Best Performing Tags:")
    for tag_info in summary["tag_performance"]["best_tags"][:5]:
        print(f"  {tag_info['tag']}: F1={tag_info['f1']:.3f} (support={tag_info['support']})")
    
    print("\nTop 5 Worst Performing Tags:")
    for tag_info in summary["tag_performance"]["worst_tags"][:5]:
        print(f"  {tag_info['tag']}: F1={tag_info['f1']:.3f} (support={tag_info['support']})")
    
    print(f"\nDetailed results saved to: {output_file}")

    return summary

def detect_model_format(model_path):
    """Detect if model is ONNX or PyTorch format."""
    if model_path.endswith('.onnx'):
        return 'onnx'
    elif model_path.endswith(('.pt', '.pth', '.bin')):
        return 'pytorch'
    return 'unknown'

def main():
    # Check GPU availability via ONNX Runtime
    providers = ort.get_available_providers()
    print(f"Available ONNX Runtime providers: {providers}")

    if 'CUDAExecutionProvider' in providers:
        print("CUDA provider available - GPU acceleration enabled")
        gpu_used, gpu_total = get_gpu_memory_info()
        if gpu_total > 0:
            print(f"GPU Memory: {gpu_used:.2f}/{gpu_total:.2f}GB")
    elif 'TensorrtExecutionProvider' in providers:
        print("TensorRT provider available - GPU acceleration enabled")
    else:
        print("WARNING: No GPU providers available, will use CPU (much slower)")

    # Check system memory
    ram_used, ram_avail = get_system_memory_info()
    print(f"System RAM: {ram_used:.2f}GB used, {ram_avail:.2f}GB available")
    print()

    data_folder = "/media/andrewk/qnap-public/workspace/shard_00022/"
    model_path = "/media/andrewk/qnap-public/workspace/OppaiOracle/exported/model.onnx"
    output_dir = "/media/andrewk/qnap-public/workspace/results"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "evaluation_results_gpu.json")

    # Adjust batch size based on your model and image size
    # For 640x640 images, 32-64 is usually good for 32GB VRAM
    # Start conservative and increase if memory allows
    BATCH_SIZE = 32  # You can increase this if memory usage is low
    MAX_MEMORY_GB = 25  # Keep within this limit

    # Detect model format and choose appropriate processing function
    model_format = detect_model_format(model_path)

    if model_format == 'pytorch':
        print("Detected PyTorch model format - using optimized PyTorch evaluation")
        process_batch_pytorch(
            data_folder=data_folder,
            model_path=model_path,
            config_path=None,
            threshold=0.5,
            image_size=640,
            output_file=output_file.replace('.json', '_pytorch.json'),
            batch_size=BATCH_SIZE,
            compile_model=True  # Enable torch.compile for extra speed
        )
    else:
        print("Using ONNX Runtime evaluation")
        process_batch_gpu(
            data_folder=data_folder,
            model_path=model_path,
            config_path=None,
            threshold=0.5,
            image_size=640,
            output_file=output_file,
            batch_size=BATCH_SIZE,
            limit=None,  # Set to a number for testing (e.g., 100)
            max_memory_gb=MAX_MEMORY_GB
        )

if __name__ == "__main__":
    main()
