#!/usr/bin/env python3
"""Convert vocabulary JSON to compressed metadata for ONNX models."""
import argparse
import json
import base64
import gzip
import hashlib
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Convert vocabulary to ONNX metadata format")
    parser.add_argument("vocab", type=str, help="Path to vocabulary.json")
    parser.add_argument("--output", type=str, help="Optional output file for metadata JSON")
    args = parser.parse_args()

    vocab_path = Path(args.vocab)
    with open(vocab_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    json_bytes = json.dumps(data, ensure_ascii=False).encode("utf-8")
    sha = hashlib.sha256(json_bytes).hexdigest()
    b64 = base64.b64encode(gzip.compress(json_bytes)).decode("utf-8")
    metadata = {"vocab_b64_gzip": b64, "vocab_sha256": sha}
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
    else:
        print(json.dumps(metadata, indent=2))

if __name__ == "__main__":
    main()
