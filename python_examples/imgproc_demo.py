#!/usr/bin/env python3
"""
Example: Image Processing in Python

This example demonstrates image processing operations available
through the cv-native Python bindings.

Requirements:
    pip install cv-native numpy

Run:
    python python_examples/imgproc_demo.py
"""

import cv_native
import numpy as np


def main():
    print("=== cv-native Image Processing Demo ===\n")

    # Create test images
    print("1. Creating test images...")
    width, height = 256, 256

    # Create grayscale image
    img = np.zeros((height, width), dtype=np.uint8)
    for y in range(height):
        for x in range(width):
            if x < width // 2 and y < height // 2:
                img[y, x] = int((x + y) / 2)
            elif x >= width // 2 and y >= height // 2:
                img[y, x] = 255 - int((x + y) / 2)
            elif y < height // 2:
                img[y, x] = int(x * 255 / width)
            else:
                img[y, x] = int(y * 255 / height)

    print(f"   Created {width}x{height} grayscale image")

    # Note: Gaussian blur requires the compiled module
    # In a full implementation, this would call:
    # blurred = cv_native.gaussian_blur(img, sigma=5.0)

    print("\n2. Image Processing Functions Available:")
    print("   - cv_native.gaussian_blur(input, sigma)")
    print("   - cv_native.detect_orb(input, n_features)")
    print("   - cv_native.match_descriptors(query, train)")

    print("\nNote: Image processing requires compiled cv_native module")
    print("      Build with: cd python && maturin develop")

    print("\n=== Demo Complete ===")


if __name__ == "__main__":
    main()
