#!/usr/bin/env python3
"""
Example: Core Types and Frame Conventions in Python

This example demonstrates the core types and frame conventions available
through the cv-native Python bindings.

Requirements:
    pip install cv-native

Run:
    python python_examples/core_demo.py
"""

import cv_native
import numpy as np


def main():
    print("=== cv-native Core Demo ===\n")

    # 1. Point Cloud
    print("1. Point Cloud:")
    xs = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
    ys = np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32)
    zs = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)

    pc = cv_native.PointCloud3D.from_arrays(xs, ys, zs)
    print(f"   Created point cloud with {pc.num_points()} points")

    # With colors
    pc.with_colors_rgb(
        np.array([1.0, 0.0, 0.0, 1.0], dtype=np.float32),
        np.array([0.0, 1.0, 0.0, 1.0], dtype=np.float32),
        np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float32),
    )
    print(f"   Added colors to point cloud")

    # With normals
    pc.with_normals(
        np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32),
        np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32),
        np.array([1.0, 0.0, 0.0, 1.0], dtype=np.float32),
    )
    print(f"   Added normals to point cloud")

    # 2. Registration
    print("\n2. Point Cloud Registration (ICP):")
    # Create two offset point clouds
    xs1 = np.array([0.0, 1.0, 1.0, 0.0], dtype=np.float32)
    ys1 = np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float32)
    zs1 = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)

    xs2 = np.array([0.1, 1.1, 1.1, 0.1], dtype=np.float32)
    ys2 = np.array([0.1, 0.1, 1.1, 1.1], dtype=np.float32)
    zs2 = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)

    source = cv_native.PointCloud3D.from_arrays(xs1, ys1, zs1)
    target = cv_native.PointCloud3D.from_arrays(xs2, ys2, zs2)

    result = cv_native.registration_icp(source, target, 1.0, 50)

    if result is not None:
        print(f"   ICP Result:")
        print(f"     Fitness: {result.fitness:.4f}")
        print(f"     RMSE: {result.inlier_rmse:.6f}")
        transform = result.get_transform()
        print(f"     Transform: {transform}")
    else:
        print("   Registration failed")

    # 3. Thread Groups (Resource Management)
    print("\n3. Thread Groups:")
    try:
        # Create a resource group
        group = cv_native.create_resource_group(
            "demo_group", num_threads=2, cores=[0, 1]
        )
        print(f"   Created group: {group.name()}")

        # Get existing group
        existing = cv_native.get_resource_group("demo_group")
        print(f"   Retrieved group: {existing.name()}")
    except Exception as e:
        print(f"   Note: Thread groups require cv-runtime: {e}")

    # 4. Rectangles and IoU
    print("\n4. Rectangle IoU:")
    r1 = cv_native.PyRect(0.0, 0.0, 10.0, 10.0)
    r2 = cv_native.PyRect(5.0, 5.0, 10.0, 10.0)
    iou = cv_native.iou(r1, r2)
    print(f"   IoU of overlapping rectangles: {iou:.4f}")

    # Vectorized IoU
    boxes1 = np.array([[0, 0, 10, 10], [20, 20, 5, 5]], dtype=np.float32)
    boxes2 = np.array([[5, 5, 10, 10], [22, 22, 5, 5]], dtype=np.float32)
    ious = cv_native.vectorized_iou(boxes1, boxes2)
    print(f"   Vectorized IoU results:\n   {ious}")

    # 5. Spatial Index
    print("\n5. Spatial Index:")
    poly1 = cv_native.PyPolygon([(0, 0), (10, 0), (10, 10), (0, 10)])
    poly2 = cv_native.PyPolygon([(5, 5), (15, 5), (15, 15), (5, 15)])
    poly3 = cv_native.PyPolygon([(20, 20), (30, 20), (30, 30), (20, 30)])

    index = cv_native.PySpatialIndex([poly1, poly2, poly3])

    # Query for overlapping polygons
    results = index.query(3, 3, 12, 12)
    print(f"   Polygons intersecting (3,3)-(12,12): {results}")

    nearest = index.nearest(7, 7)
    print(f"   Nearest to (7,7): {nearest}")

    print("\n=== Demo Complete ===")


if __name__ == "__main__":
    main()
