#!/usr/bin/env python3

"""
Download EMNIST dataset (if does not exist) and convert to JSON format for Haskell NN training.
Only needs to run once - the trained weights are committed to the repo.
"""

import gzip
import struct
import json
import random
import urllib.request
from pathlib import Path
import numpy as np


EMNIST_URL = "https://biometrics.nist.gov/cs_links/EMNIST/gzip.zip"
DATA_DIR = Path(__file__).parent.parent / "dataset"
EMNIST_DIR = DATA_DIR / "EMNIST"

def download_emnist_if_not_exists():
    """Download EMNIST dataset if not present."""
    import zipfile

    zip_path = DATA_DIR / "EMNIST.zip"
    ext_path = EMNIST_DIR

    if not DATA_DIR.exists():
        DATA_DIR.mkdir(parents=True)

    if not zip_path.exists():
        print("Downloading EMNIST dataset (this may take a while)...")
        urllib.request.urlretrieve(EMNIST_URL, zip_path)
        print("Download complete!")

    marker_file = ext_path / "emnist-letters-train-images-idx3-ubyte.gz"
    if marker_file.exists():
        print("Dataset already extracted.")
        return

    print("Extracting...")

    if not ext_path.exists():
        ext_path.mkdir(parents=True)

    with zipfile.ZipFile(zip_path, 'r') as zf:
        for member in zf.namelist():
            # Skip the root 'gzip/' folder and extract directly
            if member.startswith('gzip/'):
                # Strip 'gzip/' prefix
                filename = member[len('gzip/'):]
                if filename:  # Skip if it's just the folder itself
                    # Extract to EMNIST/ instead of EMNIST/gzip/
                    target_path = ext_path / filename
                    with zf.open(member) as source, open(target_path, 'wb') as target:
                        target.write(source.read())
    print("Extraction complete!")


def read_idx_labels(filepath):
    """Read IDX format label file."""
    with gzip.open(filepath, 'rb') as f:
        # Read 8 bytes and unpack them as two 32-bit integers
        magic, num = struct.unpack('>II', f.read(8))
        if magic != 2049:  # 0x00000801
            raise ValueError(f"Invalid magic number: {magic}, expected 2049")

        labels = list(f.read(num))
        if len(labels)!= num:
            raise ValueError(f"Expected {num} labels, got {len(labels)}")
        return labels

def read_idx_images(filepath):
    """Read IDX format image file."""
    with gzip.open(filepath, 'rb') as f:
        magic, num, rows, cols = struct.unpack('>IIII', f.read(16))
        if magic != 2051:  # 0x00000803
            raise ValueError(f"Invalid magic number: {magic}, expected 2051")

        images = np.frombuffer(f.read(), dtype=np.uint8)
        images = images.reshape(num, rows * cols)
        images = images.astype(float) / 255.0
        # Convert to list of lists as needed for JSON
        return images.tolist()

def prep_letters():
    """Prepare EMNIST Letters dataset (A-Z)."""
    base = EMNIST_DIR / "emnist-letters"

    print("  Loading letter images...")
    images = read_idx_images(f"{base}-train-images-idx3-ubyte.gz")
    print("  Loading letter labels...")
    labels = read_idx_labels(f"{base}-train-labels-idx1-ubyte.gz")

    return [
        {"pixels": img, "label": label - 1}
        for img, label in zip(images, labels)
    ]

def prep_digits():
    """Prepare EMNIST Digits dataset (0-9)."""
    base = EMNIST_DIR / "emnist-digits"

    print("  Loading digit images...")
    images = read_idx_images(f"{base}-train-images-idx3-ubyte.gz")
    print("  Loading digit labels...")
    labels = read_idx_labels(f"{base}-train-labels-idx1-ubyte.gz")

    return [
        {"pixels": img, "label": label + 26}
        for img, label in zip(images, labels)
    ]

def sample_data(data, samples_per_class=500, seed=42):
    """Take balanced sample from each class"""
    from collections import defaultdict
    import random

    by_class = defaultdict(list)
    for item in data:
        by_class[item["label"]].append(item)

    sampled = []
    random.seed(seed)  # For reproducibility
    for _, items in by_class.items():
        # Shuffle before taking samples
        random.shuffle(items)
        sampled.extend(items[:samples_per_class])

    # Shuffle final list too
    random.shuffle(sampled)
    return sampled

def main():
    output_file = Path(__file__).parent / "training_data.json"
    if output_file.exists():
        print(f"Training data already exists: {output_file}")
        return

    download_emnist_if_not_exists()

    print("Preparing letters (A-Z)...")
    letters = prep_letters()
    print(f"  Loaded {len(letters)} letter samples")

    print("Preparing digits (0-9)...")
    digits = prep_digits()
    print(f"  Loaded {len(digits)} digit samples")

    data = letters + digits
    print(f"Total samples: {len(data)}")

    sampled = sample_data(data)
    print(f"Sampled: {len(sampled)} samples")

    # Validate: should have 36 classes with ~500 samples each
    from collections import Counter
    label_counts = Counter(item["label"] for item in sampled)
    print(f"Classes: {len(label_counts)} (expected 36)")
    print(f"Samples per class: min={min(label_counts.values())}, max={max(label_counts.values())}")

    print(f"Saving to {output_file}...")
    with open(output_file, 'w') as f:
        json.dump(sampled, f, separators=(',', ':'))

    print(f"File size: {output_file.stat().st_size / 1024 / 1024:.2f} MB")
    print("Done!")

if __name__ == "__main__":
    main()
