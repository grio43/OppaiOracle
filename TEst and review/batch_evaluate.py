 #!/usr/bin/env python3
import argparse, json, os, sys, time, pathlib
from collections import defaultdict
from typing import List, Tuple, Dict, Set
import numpy as np
from PIL import Image
import onnxruntime as ort
from tqdm import tqdm

def load_tag_names(config_path):
    if config_path and os.path.exists(config_path):
        with open(config_path, "r") as f:
            cfg = json.load(f)
        names = cfg.get("tag_names", None)
        norm = cfg.get("normalization_params", None)
        return names, norm
    return None, None

def preprocess(image_path, image_size=640, mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5)):
    img = Image.open(image_path).convert("RGB").resize((image_size, image_size))
    arr = np.array(img).astype("float32")/255.0
    # HWC -> CHW
    arr = arr.transpose(2,0,1)
    # normalize
    mean = np.array(mean, dtype="float32")[:,None,None]
    std  = np.array(std,  dtype="float32")[:,None,None]
    arr = (arr - mean)/std
    # add batch
    return np.expand_dims(arr, 0)

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
    # Split tags and clean them
    tags = set(tag.strip() for tag in tags_str.split() if tag.strip())
    
    # Also check for rating if it's a tag
    rating = json_data.get("rating", "")
    if rating and rating != "safe":  # You might want to include/exclude certain ratings
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
        "false_negatives": false_negatives
    }

def process_batch(data_folder, model_path, config_path=None, threshold=0.5, 
                  image_size=640, providers=None, output_file="evaluation_results.json",
                  limit=None):
    """Process all images in the folder and evaluate against ground truth"""
    
    # Load model configuration
    tag_names, norm = load_tag_names(config_path)
    mean, std = (0.5,0.5,0.5), (0.5,0.5,0.5)
    if norm and "mean" in norm and "std" in norm:
        mean, std = tuple(norm["mean"]), tuple(norm["std"])
    
    # Initialize ONNX session
    providers = providers or ort.get_available_providers()
    sess = ort.InferenceSession(model_path, providers=providers)
    
    # Determine output name
    output_names = [o.name for o in sess.get_outputs()]
    use_scores = "scores" if "scores" in output_names else output_names[0]
    
    # Find all JSON files
    json_files = list(pathlib.Path(data_folder).glob("*.json"))
    if limit:
        json_files = json_files[:limit]
    
    print(f"Found {len(json_files)} JSON files to process")
    
    # Results storage
    all_results = []
    overall_metrics = defaultdict(list)
    tag_performance = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    failed_files = []
    
    # Process each file
    for json_path in tqdm(json_files, desc="Processing images"):
        try:
            # Load JSON data
            with open(json_path, "r") as f:
                json_data = json.load(f)
            
            # Get image path
            image_filename = json_data.get("filename")
            if not image_filename:
                print(f"Warning: No filename in {json_path}")
                continue
            
            image_path = json_path.parent / image_filename
            if not image_path.exists():
                print(f"Warning: Image not found: {image_path}")
                failed_files.append(str(json_path))
                continue
            
            # Get ground truth tags
            ground_truth = parse_ground_truth_tags(json_data)
            
            # Run inference
            x = preprocess(str(image_path), image_size, mean, std)
            inputs = {"input_image": x.astype("float32")}
            
            start_time = time.time()
            outs = sess.run([use_scores], inputs)
            inference_time = time.time() - start_time
            
            scores = outs[0]
            
            # Get predictions
            predicted_tags = get_predictions_from_scores(scores, tag_names, threshold)
            
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
                "filename": image_filename,
                "ground_truth_tags": list(ground_truth),
                "predicted_tags": list(predicted_tags),
                "metrics": metrics,
                "inference_time": inference_time,
                "num_gt_tags": len(ground_truth),
                "num_pred_tags": len(predicted_tags)
            }
            all_results.append(result)
            
            # Update overall metrics
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    overall_metrics[key].append(value)
            
        except Exception as e:
            print(f"Error processing {json_path}: {e}")
            failed_files.append(str(json_path))
            continue
    
    # Calculate aggregate statistics
    summary = {
        "total_files": len(json_files),
        "processed_files": len(all_results),
        "failed_files": len(failed_files),
        "threshold": threshold,
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
            "support": tp + fn  # Number of ground truth occurrences
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
    print(f"Threshold: {threshold}")
    print("\nAverage Metrics:")
    for metric, value in summary["average_metrics"].items():
        if not metric.endswith("_std"):
            print(f"  {metric}: {value:.4f}")
    
    print("\nTop 5 Best Performing Tags:")
    for tag_info in summary["tag_performance"]["best_tags"][:5]:
        print(f"  {tag_info['tag']}: F1={tag_info['f1']:.3f} (support={tag_info['support']})")
    
    print("\nTop 5 Worst Performing Tags:")
    for tag_info in summary["tag_performance"]["worst_tags"][:5]:
        print(f"  {tag_info['tag']}: F1={tag_info['f1']:.3f} (support={tag_info['support']})")
    
    print(f"\nDetailed results saved to: {output_file}")
    
    return summary

def main():
    ap = argparse.ArgumentParser(description="Batch evaluation of model against training data")
    ap.add_argument("--data_folder", required=True, help="Path to folder containing JSON and image files")
    ap.add_argument("--model", default="exported_models/model.onnx", help="Path to ONNX model")
    ap.add_argument("--config", default="./checkpoints/model_config.json", help="Optional model_config.json")
    ap.add_argument("--threshold", type=float, default=0.5, help="Confidence threshold for predictions")
    ap.add_argument("--image_size", type=int, default=640, help="Image size for preprocessing")
    ap.add_argument("--output", default="evaluation_results.json", help="Output file for results")
    ap.add_argument("--limit", type=int, default=None, help="Limit number of files to process (for testing)")
    ap.add_argument("--providers", nargs="*", default=None,
                    help="Override ONNX Runtime providers, e.g. CUDAExecutionProvider")
    
    args = ap.parse_args()
    
    process_batch(
        data_folder=args.data_folder,
        model_path=args.model,
        config_path=args.config,
        threshold=args.threshold,
        image_size=args.image_size,
        providers=args.providers,
        output_file=args.output,
        limit=args.limit
    )

if __name__ == "__main__":
    main()