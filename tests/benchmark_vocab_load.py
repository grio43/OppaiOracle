#!/usr/bin/env python3
"""Quick benchmark for JSON vocabulary loading performance."""

import time
from pathlib import Path

def benchmark_json():
    """Benchmark JSON vocabulary loading."""
    from vocabulary import load_vocabulary_for_training

    vocab_path = Path("vocabulary.json")
    if not vocab_path.exists():
        return None

    start = time.perf_counter()
    vocab = load_vocabulary_for_training(vocab_path)
    elapsed = time.perf_counter() - start

    return elapsed, len(vocab.tag_to_index)

if __name__ == "__main__":
    print("JSON Vocabulary Loading Benchmark\n" + "="*50)

    # Run multiple times to get average
    json_times = []

    for i in range(5):
        result = benchmark_json()
        if result:
            elapsed, size = result
            json_times.append(elapsed)
            if i == 0:
                print(f"\nVocabulary size: {size:,} tags")
                print(f"File format: JSON")

    if json_times:
        avg_json = sum(json_times) / len(json_times)
        min_time = min(json_times)
        max_time = max(json_times)

        print(f"\nAverage load time: {avg_json*1000:.2f} ms")
        print(f"Min load time: {min_time*1000:.2f} ms")
        print(f"Max load time: {max_time*1000:.2f} ms")

        print(f"\nOver 100 restarts (dev/debug): {avg_json*100:.2f} seconds")
        print(f"Per-epoch restart overhead: ~{avg_json*1000:.0f} ms")
    else:
        print("\nNo vocabulary file found. Please ensure vocabulary.json exists.")
