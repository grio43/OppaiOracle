"""
Analyze tag distribution in the first dataset shard.
Calculates median, mean, and percentile statistics for tag counts per image.
"""

import json
import os
from pathlib import Path
import statistics

def parse_tags(tags_str):
    """Parse comma-separated tags, matching dataset loader logic."""
    if not tags_str or not isinstance(tags_str, str):
        return []

    # Split by comma, strip whitespace, filter empty strings
    tags = [tag.strip() for tag in tags_str.split(',')]
    tags = [tag for tag in tags if tag]
    return tags

def analyze_shard(shard_path, show_details=True):
    """Analyze tag counts in a dataset shard."""
    shard_dir = Path(shard_path)

    if not shard_dir.exists():
        print(f"Error: Shard directory not found at {shard_path}")
        return None

    tag_counts = []
    json_files = list(shard_dir.glob("*.json"))

    if show_details:
        print(f"Found {len(json_files)} JSON files in {shard_path}")
        print("Analyzing tag counts...")

    errors = 0
    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            tags_str = data.get('tags', '')
            tags = parse_tags(tags_str)
            tag_counts.append(len(tags))

        except Exception as e:
            errors += 1
            if errors <= 5 and show_details:  # Only print first few errors
                print(f"Error reading {json_file}: {e}")

    if errors > 5 and show_details:
        print(f"... and {errors - 5} more errors")

    if not tag_counts:
        print("No valid tag data found!")
        return None

    return {
        'tag_counts': tag_counts,
        'total': len(tag_counts),
        'errors': errors
    }

def calculate_statistics(all_tag_counts):
    """Calculate statistics from aggregated tag counts."""
    if not all_tag_counts:
        print("No tag counts to analyze!")
        return None

    # Sort for percentile calculations
    all_tag_counts.sort()

    median = statistics.median(all_tag_counts)
    mean = statistics.mean(all_tag_counts)
    min_tags = min(all_tag_counts)
    max_tags = max(all_tag_counts)

    # Calculate percentiles
    def percentile(data, p):
        """Calculate the p-th percentile of sorted data."""
        if not data:
            return 0
        k = (len(data) - 1) * p / 100
        f = int(k)
        c = f + 1
        if c >= len(data):
            return data[-1]
        d0 = data[f]
        d1 = data[c]
        return d0 + (d1 - d0) * (k - f)

    p5 = percentile(all_tag_counts, 5)
    p10 = percentile(all_tag_counts, 10)
    p25 = percentile(all_tag_counts, 25)
    p50 = percentile(all_tag_counts, 50)
    p75 = percentile(all_tag_counts, 75)
    p90 = percentile(all_tag_counts, 90)
    p95 = percentile(all_tag_counts, 95)
    p99 = percentile(all_tag_counts, 99)

    # Print results
    print("\n" + "="*60)
    print("COMBINED TAG COUNT STATISTICS (ALL SHARDS)")
    print("="*60)
    print(f"Total images analyzed: {len(all_tag_counts):,}")
    print()
    print(f"Median tag count:      {median:.1f}")
    print(f"Mean tag count:        {mean:.2f}")
    print(f"Min tag count:         {min_tags}")
    print(f"Max tag count:         {max_tags}")
    print()
    print("Distribution percentiles:")
    print(f"   5th percentile:     {p5:.1f}")
    print(f"  10th percentile:     {p10:.1f}")
    print(f"  25th percentile:     {p25:.1f}")
    print(f"  50th percentile:     {p50:.1f} (median)")
    print(f"  75th percentile:     {p75:.1f}")
    print(f"  90th percentile:     {p90:.1f}")
    print(f"  95th percentile:     {p95:.1f}")
    print(f"  99th percentile:     {p99:.1f}")
    print("="*60)

    # Additional insights for filtering
    print("\nInsights for filtering poorly tagged images:")
    print(f"\nImages with different thresholds:")
    for threshold in [5, 10, 15, 20, 25, 30]:
        count = sum(1 for x in all_tag_counts if x < threshold)
        pct = count / len(all_tag_counts) * 100
        print(f"  <{threshold:2d} tags: {count:6,} images ({pct:5.2f}%)")

    return {
        'total': len(all_tag_counts),
        'median': median,
        'mean': mean,
        'min': min_tags,
        'max': max_tags,
        'percentiles': {
            'p5': p5,
            'p10': p10,
            'p25': p25,
            'p50': p50,
            'p75': p75,
            'p90': p90,
            'p95': p95,
            'p99': p99
        }
    }

if __name__ == "__main__":
    # Analyze multiple shards for larger sample
    base_path = r"L:\Dab\Dab"
    shard_numbers = [0, 1, 2, 3]  # Analyze first 4 shards

    all_tag_counts = []
    total_errors = 0

    print("Analyzing multiple shards for comprehensive statistics...")
    print("="*60)

    for shard_num in shard_numbers:
        shard_path = os.path.join(base_path, f"shard_{shard_num:05d}")
        print(f"\nProcessing shard_{shard_num:05d}...")

        result = analyze_shard(shard_path, show_details=False)

        if result:
            all_tag_counts.extend(result['tag_counts'])
            total_errors += result['errors']
            print(f"  Analyzed {result['total']:,} images ({result['errors']} errors)")
        else:
            print(f"  Failed to analyze shard_{shard_num:05d}")

    print("\n" + "="*60)
    print(f"Total errors across all shards: {total_errors}")

    # Calculate and display combined statistics
    if all_tag_counts:
        calculate_statistics(all_tag_counts)
    else:
        print("No tag data collected from any shard!")
