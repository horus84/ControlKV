#!/usr/bin/env python3
"""
Build fixed 150-example BFCL benchmark subset and save dataset_subset.json with SHA-256 hash.
"""
import sys
import json
from controlkv.benchmarks.subset import load_or_create_bfcl_subset

def main():
    print("Building 150-example BFCL benchmark subset...")
    manifest = load_or_create_bfcl_subset("results/dataset_subset.json")
    print(f"Dataset Name: {manifest['dataset_name']}")
    print(f"Total Examples: {manifest['num_examples']}")
    print(f"Categories Breakdown: {manifest['categories']}")
    print(f"SHA-256 Content Hash: {manifest['content_sha256']}")
    print("Saved to results/dataset_subset.json")

if __name__ == "__main__":
    main()
