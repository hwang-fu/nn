#!/usr/bin/env python3

"""
Download EMNIST dataset (if does not exist) and convert to JSON format for Haskell NN training.
Only needs to run once - the trained weights are committed to the repo.
"""

import gzip
import json
import os
import struct
import urllib.request
from pathlib import Path


EMNIST_URL = "https://biometrics.nist.gov/cs_links/EMNIST/gzip.zip"
DATA_DIR = Path(__file__).parent / "dataset" / "EMNIST"

def download_emnist_if_not_exist():
    """Download EMNIST dataset if not present."""

    if DATA_DIR.exists():
        return

    import zipfile

    zip_path = DATA_DIR / "EMNIST.zip"

    if not DATA_DIR.exists():
        DATA_DIR.mkdir(parents=True)

    if not zip_path.exists():
        print("Downloading EMNIST dataset (this may take a while)...")
        urllib.request.urlretrieve(EMNIST_URL, zip_path)
        print("Download complete!")

    print("Extracting...")
    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(DATA_DIR)
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

        images = []
        for _ in range(num):
            img = list(f.read(rows * cols))
            # normalize to 0 ~ 1
            images.append([p / 255.0 for p in img])
        return images

def prep_letters():
     """Prepare EMNIST Letters dataset (A-Z)."""

def prep_digits():
    """Prepare EMNIST Digits dataset (0-9)."""
    base = DATA_DIR / "EMNIST" / "emnist-digits"

    images = read_idx_images(f"{base}-train-images-idx3-ubyte.gz")
    labels = read_idx_labels(f"{base}-train-labels-idx1-ubyte.gz")

    data = []
    for img, label = in zip(images, labels):
        data.append()


def main():
    pass

if __name__ == "__main__":
    main()
