#!/usr/bin/env python3
import json, os, sys, time, pathlib
from collections import defaultdict
from typing import List, Tuple, Dict, Set
import numpy as np
from PIL import Image
import onnxruntime as ort
from tqdm import tqdm
import gc
import psutil  # For system memory monitoring


def get_system_memory_info():
    """Get current system memory usage"""
    mem = psutil.virtual_memory()
    return mem.used / 1024 ** 3, mem.available / 1024 ** 3  # GB


def get_gpu_memory_info():
    """Get GPU memory info using nvidia-ml-py if available, otherwise nvidia-smi"""
    try:
        import pynvml

        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        used = info.used / 1024 ** 3  # GB
        total = info.total / 1024 ** 3  # GB
        return used, total
    except Exception:
        # Fallback: parse nvidia-smi output
        import subprocess

        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=memory.used,memory.total",
                    "--format=csv,nounits,noheader",
                ],
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                used, total = map(float, result.stdout.strip().split(","))
                return used / 1024, total / 1024  # Convert MB to GB
        except Exception:
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


def preprocess_batch(
    image_paths,
    image_size=640,
    mean=(0.5, 0.5, 0.5),
    std=(0.5, 0.5, 0.5),
    gray_background=(114, 114, 114),
):
    """Preprocess multiple images into a batch with proper transparency handling and padding"""
    batch = []
    valid_indices = []

    for idx, image_path in enumerate(image_paths):
        try:
            img = Image.open(image_path)

            # Handle transparency by compositing on gray background
            if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
                background = Image.new("RGB", img.size, gray_background)
                if img.mode != "RGBA":
                    img = img.convert("RGBA")
                background.paste(img, mask=img.split()[3])
                img = background
            else:
                img = img.convert("RGB")

            # Resize with padding to maintain aspect ratio
            img.thumbnail((image_size, image_size), Image.Resampling.LANCZOS)

            if img.size != (image_size, image_size):
                padded = Image.new("RGB", (image_size, image_size), gray_background)
                x = (image_size - img.width) // 2
                y = (image_size - img.height) // 2
                padded.paste(img, (x, y))
                img = padded

            arr = np.array(img, dtype=np.float32) / 255.0
            arr = arr.transpose(2, 0, 1)
            mean_arr = np.array(mean, dtype=np.float32)[:, None, None]
            std_arr = np.array(std, dtype=np.float32)[:, None, None]
            arr = (arr - mean_arr) / std_arr
            batch.append(arr)
            valid_indices.append(idx)
        except Exception as e:
            print(f"Error processing image {image_path}: {e}")
            continue

    if batch:
        return np.stack(batch, axis=0).astype(np.float32), valid_indices
    return None, []


def get_predictions_from_scores(scores, tag_names, threshold=0.5):
    """Get predicted tags above threshold"""
    s = scores[0] if scores.ndim == 2 else scores
    predicted_tags = set()
    for i, score in enumerate(s):
        if score >= threshold:
            tag = tag_names[i] if tag_names and i < len(tag_names) else f"tag_{i}"
            predicted_tags.add(tag)
    return predicted_tags


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
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    if not predicted:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    if not ground_truth:
        return {"precision": 0.0, "recall": 1.0, "f1": 0.0}

    true_positives = len(predicted & ground_truth)
    false_positives = len(predicted - ground_truth)
    false_negatives = len(ground_truth - predicted)

    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
    }


def create_session(model_path, providers=None, max_memory_gb=12):
    """Create an ONNX Runtime session with custom options"""
    providers = providers or ort.get_available_providers()
    provider_options = None
    if "CUDAExecutionProvider" in providers:
        provider_options = {
            "device_id": 0,
            "arena_extend_strategy": "kSameAsRequested",
            "gpu_mem_limit": int(max_memory_gb * 1024 * 1024 * 1024),
            "cudnn_conv_algo_search": "HEURISTIC",
            "do_copy_in_default_stream": True,
            "cudnn_conv_use_max_workspace": True,
            "enable_cuda_graph": False,
        }

    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    sess_options.enable_mem_pattern = False
    sess_options.inter_op_num_threads = 4
    sess_options.intra_op_num_threads = 4

    if provider_options:
        session = ort.InferenceSession(
            model_path,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
            provider_options=[provider_options, {}],
            sess_options=sess_options,
        )
    else:
        session = ort.InferenceSession(model_path, providers=providers, sess_options=sess_options)

    actual_provider = session.get_providers()[0]
    print(f"Session created with provider: {actual_provider}")

    try:
        dummy_input = np.random.randn(1, 3, 640, 640).astype(np.float32)
        _ = session.run(None, {session.get_inputs()[0].name: dummy_input})
        print("Warmup run completed")
    except Exception as e:
        print(f"Warmup failed (non-critical): {e}")

    return session


def process_batch(
    data_folder,
    model_path,
    config_path=None,
    threshold=0.5,
    image_size=640,
    batch_size=16,
    providers=None,
    output_file="evaluation_results.json",
    limit=None,
    max_memory_gb=12,
):
    """Process all images in the folder and evaluate against ground truth"""

    tag_names, norm = load_tag_names(config_path)
    mean, std = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
    if norm and "mean" in norm and "std" in norm:
        mean, std = tuple(norm["mean"]), tuple(norm["std"])

    sess = create_session(model_path, providers, max_memory_gb=max_memory_gb)

    output_names = [o.name for o in sess.get_outputs()]
    use_scores = "scores" if "scores" in output_names else output_names[0]

    input_names = [i.name for i in sess.get_inputs()]
    input_name = input_names[0] if input_names else "input_image"

    json_files = list(pathlib.Path(data_folder).glob("*.json"))
    if limit:
        json_files = json_files[:limit]

    print(f"Found {len(json_files)} JSON files to process")

    all_results = []
    overall_metrics = defaultdict(list)
    tag_performance = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    failed_files = []

    total_batches = (len(json_files) + batch_size - 1) // batch_size
    total_inference_time = 0

    ram_used_init, ram_avail_init = get_system_memory_info()
    gpu_used_init, gpu_total_init = get_gpu_memory_info()
    print(
        f"Initial RAM: {ram_used_init:.2f}GB used, {ram_avail_init:.2f}GB available"
    )
    print(
        f"Initial GPU: {gpu_used_init:.2f}GB used / {gpu_total_init:.2f}GB total"
    )

    processed_count = 0

    for batch_start in tqdm(
        range(0, len(json_files), batch_size), desc="Processing images"
    ):
        batch_files = json_files[batch_start : batch_start + batch_size]
        batch_image_paths = []
        batch_ground_truths = []
        valid_json_paths = []

        for json_path in batch_files:
            try:
                with open(json_path, "r") as f:
                    json_data = json.load(f)
                image_filename = json_data.get("filename")
                if not image_filename:
                    print(f"Warning: No filename in {json_path}")
                    continue
                image_path = json_path.parent / image_filename
                if not image_path.exists():
                    print(f"Warning: Image not found: {image_path}")
                    failed_files.append(str(json_path))
                    continue
                batch_image_paths.append(str(image_path))
                batch_ground_truths.append(parse_ground_truth_tags(json_data))
                valid_json_paths.append(json_path)
            except Exception as e:
                print(f"Error processing {json_path}: {e}")
                failed_files.append(str(json_path))
                continue

        batch_input, valid_indices = preprocess_batch(
            batch_image_paths, image_size, mean, std, gray_background=(114, 114, 114)
        )
        if batch_input is None:
            continue

        try:
            inputs = {input_name: batch_input}

            if processed_count % 100 == 0:
                ram_used, ram_avail = get_system_memory_info()
                gpu_used, gpu_total = get_gpu_memory_info()
                print(
                    f"\n[After {processed_count} images] RAM: {ram_used:.2f}GB used, GPU: {gpu_used:.2f}/{gpu_total:.2f}GB"
                )

            start_time = time.time()
            outs = sess.run([use_scores], inputs)
            batch_inference_time = time.time() - start_time
            total_inference_time += batch_inference_time

            scores = outs[0]

            for out_idx, img_idx in enumerate(valid_indices):
                ground_truth = batch_ground_truths[img_idx]
                image_filename = os.path.basename(batch_image_paths[img_idx])

                predicted_tags = get_predictions_from_scores(
                    scores[out_idx], tag_names, threshold
                )
                metrics = calculate_metrics(predicted_tags, ground_truth)
                result = {
                    "filename": image_filename,
                    "ground_truth_tags": list(ground_truth),
                    "predicted_tags": list(predicted_tags),
                    "metrics": metrics,
                    "inference_time": batch_inference_time / len(valid_indices),
                    "num_gt_tags": len(ground_truth),
                    "num_pred_tags": len(predicted_tags),
                }
                all_results.append(result)
                processed_count += 1

                for tag in predicted_tags & ground_truth:
                    tag_performance[tag]["tp"] += 1
                for tag in predicted_tags - ground_truth:
                    tag_performance[tag]["fp"] += 1
                for tag in ground_truth - predicted_tags:
                    tag_performance[tag]["fn"] += 1

                for key, value in metrics.items():
                    if isinstance(value, (int, float)):
                        overall_metrics[key].append(value)

            del batch_input, scores, outs

            if len(all_results) % 500 == 0:
                with open(
                    output_file.replace(
                        ".json", f"_partial_{len(all_results)}.json"
                    ),
                    "w",
                ) as f:
                    json.dump({"partial_results": all_results[-500:]}, f)
                gc.collect()

        except Exception as e:
            print(f"Error during inference for batch starting {batch_start}: {e}")
            failed_files.extend([str(p) for p in batch_files])
            continue

    total_images = len(all_results)
    avg_inference_time = (
        total_inference_time / total_images if total_images else 0
    )
    images_per_second = (
        total_images / total_inference_time if total_inference_time > 0 else 0
    )

    summary = {
        "total_files": len(json_files),
        "processed_files": len(all_results),
        "failed_files": len(failed_files),
        "threshold": threshold,
        "average_metrics": {},
        "tag_performance": {},
        "failed_file_list": failed_files,
        "total_inference_time": total_inference_time,
        "average_inference_time": avg_inference_time,
        "images_per_second": images_per_second,
    }

    for metric, values in overall_metrics.items():
        if values:
            summary["average_metrics"][metric] = float(np.mean(values))
            summary["average_metrics"][f"{metric}_std"] = float(np.std(values))

    tag_f1_scores = []
    for tag, counts in tag_performance.items():
        tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = (
            2 * (precision * recall) / (precision + recall)
            if (precision + recall) > 0
            else 0
        )
        tag_f1_scores.append(
            {
                "tag": tag,
                "f1": f1,
                "precision": precision,
                "recall": recall,
                "support": tp + fn,
            }
        )

    tag_f1_scores.sort(key=lambda x: x["f1"], reverse=True)
    summary["tag_performance"] = {
        "best_tags": tag_f1_scores[:10],
        "worst_tags": tag_f1_scores[-10:] if len(tag_f1_scores) > 10 else [],
    }

    with open(output_file, "w") as f:
        json.dump({"summary": summary, "detailed_results": all_results}, f, indent=2)

    print("\n" + "=" * 50)
    print("EVALUATION SUMMARY")
    print("=" * 50)
    print(f"Processed: {summary['processed_files']}/{summary['total_files']} files")
    print(f"Failed: {summary['failed_files']} files")
    print(f"Threshold: {threshold}")
    print(f"Total Inference Time: {total_inference_time:.2f}s")
    print(f"Images per second: {summary['images_per_second']:.2f}")

    ram_used_final, ram_avail_final = get_system_memory_info()
    gpu_used_final, gpu_total_final = get_gpu_memory_info()
    print(
        f"\nFinal RAM: {ram_used_final:.2f}GB used (Δ{ram_used_final - ram_used_init:+.2f}GB)"
    )
    print(
        f"Final GPU: {gpu_used_final:.2f}/{gpu_total_final:.2f}GB used (Δ{gpu_used_final - gpu_used_init:+.2f}GB)"
    )

    print("\nAverage Metrics:")
    for metric, value in summary["average_metrics"].items():
        if not metric.endswith("_std"):
            print(f"  {metric}: {value:.4f}")

    print("\nTop 5 Best Performing Tags:")
    for tag_info in summary["tag_performance"]["best_tags"][:5]:
        print(
            f"  {tag_info['tag']}: F1={tag_info['f1']:.3f} (support={tag_info['support']})"
        )

    print("\nTop 5 Worst Performing Tags:")
    for tag_info in summary["tag_performance"]["worst_tags"][:5]:
        print(
            f"  {tag_info['tag']}: F1={tag_info['f1']:.3f} (support={tag_info['support']})"
        )

    print(f"\nDetailed results saved to: {output_file}")

    return summary


def main():
    providers = ort.get_available_providers()
    print(f"Available ONNX Runtime providers: {providers}")

    if "CUDAExecutionProvider" in providers:
        print("CUDA provider available - GPU acceleration enabled")
        gpu_used, gpu_total = get_gpu_memory_info()
        if gpu_total > 0:
            print(f"GPU Memory: {gpu_used:.2f}/{gpu_total:.2f}GB")
    elif "TensorrtExecutionProvider" in providers:
        print("TensorRT provider available - GPU acceleration enabled")
    else:
        print("WARNING: No GPU providers available, will use CPU (much slower)")

    ram_used, ram_avail = get_system_memory_info()
    print(f"System RAM: {ram_used:.2f}GB used, {ram_avail:.2f}GB available")
    print()

    data_folder = "/media/andrewk/qnap-public/workspace/shard_00022/"
    model_path = "/media/andrewk/qnap-public/workspace/OppaiOracle/exported/model.onnx"
    output_dir = "/media/andrewk/qnap-public/workspace/results"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "evaluation_results.json")

    process_batch(
        data_folder=data_folder,
        model_path=model_path,
        config_path=None,
        threshold=0.5,
        image_size=640,
        batch_size=16,
        providers=None,
        output_file=output_file,
        limit=None,
    )


if __name__ == "__main__":
    main()
