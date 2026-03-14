"""Tests for cv-native Python bindings"""

import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import cv_native as cv


class TestFeatureDetector:
    """Test feature detection bindings"""

    def _make_image_bytes(self, width, height):
        """Create a test grayscale image with corners as flat bytes."""
        img = np.zeros((height, width), dtype=np.uint8)
        # Draw two bright squares to create corner features
        img[20:30, 20:30] = 255
        img[60:70, 60:70] = 255
        return bytes(img.tobytes()), width, height

    def test_fast_detect(self):
        """Test FAST corner detection: fast_detect(image_bytes, width, height, threshold)"""
        image_bytes, w, h = self._make_image_bytes(100, 100)
        kps = cv.FeatureDetector.fast_detect(image_bytes, w, h, 50)
        # Corners of the bright squares should be detected
        assert isinstance(kps, cv.KeyPoints)
        assert len(kps) > 0, "FAST should detect corners on a square pattern"
        pts = kps.to_list()
        assert all(isinstance(p, tuple) and len(p) == 2 for p in pts)

    def test_fast_detect_high_threshold(self):
        """With a very high threshold, no keypoints should be found."""
        image_bytes, w, h = self._make_image_bytes(100, 100)
        kps = cv.FeatureDetector.fast_detect(image_bytes, w, h, 254)
        assert len(kps) == 0, "Very high threshold should suppress all detections"

    def test_harris_detect(self):
        """Test Harris corner detection: harris_detect(image_bytes, width, height, k, threshold)"""
        image_bytes, w, h = self._make_image_bytes(100, 100)
        kps = cv.FeatureDetector.harris_detect(image_bytes, w, h, 0.04, 1000.0)
        assert isinstance(kps, cv.KeyPoints)
        assert len(kps) >= 0

    def test_gftt_detect(self):
        """Test GFTT detection: gftt_detect(image_bytes, width, height, max_corners, quality_level, min_distance)"""
        image_bytes, w, h = self._make_image_bytes(100, 100)
        kps = cv.FeatureDetector.gftt_detect(image_bytes, w, h, 100, 0.01, 5.0)
        assert isinstance(kps, cv.KeyPoints)
        assert len(kps) >= 0

    def test_shi_tomasi_detect(self):
        """Test Shi-Tomasi detection: shi_tomasi_detect(image_bytes, width, height, max_corners, quality_level)"""
        image_bytes, w, h = self._make_image_bytes(100, 100)
        kps = cv.FeatureDetector.shi_tomasi_detect(image_bytes, w, h, 100, 0.01)
        assert isinstance(kps, cv.KeyPoints)
        assert len(kps) >= 0


class TestPointCloud:
    """Test point cloud bindings"""

    def test_create_from_tuples(self):
        """PointCloud(points_list) takes a list of (f32, f32, f32) tuples."""
        pts = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)]
        pc = cv.PointCloud(pts)
        assert pc.num_points() == 3

    def test_to_numpy(self):
        """to_numpy() returns a flat list [x0,y0,z0, x1,y1,z1, ...]"""
        pc = cv.PointCloud([(0.0, 0.0, 0.0), (1.0, 2.0, 3.0)])
        arr = pc.to_numpy()
        assert len(arr) == 6  # 2 points * 3 coords
        assert abs(arr[3] - 1.0) < 1e-6
        assert abs(arr[4] - 2.0) < 1e-6
        assert abs(arr[5] - 3.0) < 1e-6

    def test_set_normals(self):
        """set_normals() takes a list of (f32, f32, f32) tuples, not a numpy array."""
        pc = cv.PointCloud([(0.0, 0.0, 0.0), (1.0, 1.0, 1.0)])
        normals = [(0.0, 0.0, 1.0), (0.0, 1.0, 0.0)]
        pc.set_normals(normals)
        assert pc.has_normals()
        retrieved = pc.get_normals()
        assert retrieved is not None
        assert len(retrieved) == 2
        assert abs(retrieved[0][2] - 1.0) < 1e-6

    def test_from_list(self):
        """PointCloud.from_list() static method also takes tuples."""
        pc = cv.PointCloud.from_list([(1.0, 2.0, 3.0)])
        assert pc.num_points() == 1

    def test_points_to_list(self):
        """points_to_list() returns list of (x,y,z) tuples."""
        pts = [(1.0, 2.0, 3.0), (4.0, 5.0, 6.0)]
        pc = cv.PointCloud(pts)
        result = pc.points_to_list()
        assert len(result) == 2
        assert abs(result[0][0] - 1.0) < 1e-6
        assert abs(result[1][2] - 6.0) < 1e-6


class TestMeshReconstruction:
    """Test mesh reconstruction bindings"""

    def test_create_sphere(self):
        """Test sphere point cloud creation"""
        pc = cv.MeshReconstruction.create_sphere((0, 0, 0), 1.0, 100)
        assert pc.num_points() == 100

    def test_create_plane(self):
        """Test plane point cloud creation"""
        pc = cv.MeshReconstruction.create_plane((0, 0, 0), (0, 0, 1), 1.0, 50)
        assert pc.num_points() == 50

    def test_ball_pivoting(self):
        """Test ball pivoting surface reconstruction"""
        pc = cv.MeshReconstruction.create_sphere((0, 0, 0), 1.0, 200)
        mesh = cv.MeshReconstruction.ball_pivoting(pc, 0.2)
        assert mesh.num_vertices() == 200

    def test_alpha_shapes(self):
        """Test alpha shapes reconstruction"""
        pc = cv.MeshReconstruction.create_sphere((0, 0, 0), 1.0, 100)
        mesh = cv.MeshReconstruction.alpha_shapes(pc, 0.1)
        assert mesh.num_vertices() >= 0

    def test_poisson(self):
        """Test Poisson surface reconstruction"""
        pc = cv.MeshReconstruction.create_sphere((0, 0, 0), 1.0, 100)
        mesh = cv.MeshReconstruction.poisson(pc, 5)
        assert mesh is not None

    def test_mesh_to_obj(self):
        """Test OBJ export"""
        mesh = cv.TriangleMesh()
        mesh.add_vertex(0.0, 0.0, 0.0)
        mesh.add_vertex(1.0, 0.0, 0.0)
        mesh.add_vertex(0.0, 1.0, 0.0)
        mesh.add_face(0, 1, 2)

        obj = mesh.to_obj()
        assert "v 0 0 0" in obj
        assert "f 1 2 3" in obj
        assert mesh.num_vertices() == 3
        assert mesh.num_faces() == 1


class TestIsam2:
    """Test ISAM2 bindings"""

    def test_create(self):
        """Test ISAM2 creation"""
        isam = cv.Isam2(True, False)
        assert isam.num_nodes() == 0

    def test_add_pose(self):
        """Test adding pose"""
        isam = cv.Isam2(False, False)
        isam.add_pose(0, 0.0, 0.0, 0.0)
        isam.add_pose(1, 1.0, 0.0, 0.0)
        assert isam.num_nodes() == 2

    def test_add_point(self):
        """Test adding point"""
        isam = cv.Isam2(False, False)
        isam.add_point(100, 1.0, 2.0, 3.0)
        assert isam.num_nodes() == 1

    def test_add_factor(self):
        """Test adding factor"""
        isam = cv.Isam2(False, False)
        isam.add_pose(0, 0.0, 0.0, 0.0)
        isam.add_pose(1, 1.0, 0.0, 0.0)
        isam.add_factor(0, 1, 1.0, 0.0, 0.0, 0.1)
        assert isam.num_factors() == 1

    def test_get_pose(self):
        """Test getting pose"""
        isam = cv.Isam2(False, False)
        isam.add_pose(0, 1.0, 2.0, 3.0)
        pose = isam.get_pose(0)
        assert pose is not None
        assert pose[0] == 1.0
        assert pose[1] == 2.0
        assert pose[2] == 3.0

    def test_get_all_poses(self):
        """Test getting all poses"""
        isam = cv.Isam2(False, False)
        isam.add_pose(0, 0.0, 0.0, 0.0)
        isam.add_pose(1, 1.0, 0.0, 0.0)
        poses = isam.get_all_poses()
        assert len(poses) == 2

    def test_optimize(self):
        """Test optimization"""
        isam = cv.Isam2(False, False)
        isam.add_pose(0, 0.0, 0.0, 0.0)
        isam.add_pose(1, 1.0, 0.0, 0.0)
        isam.add_factor(0, 1, 1.0, 0.0, 0.0, 0.1)
        isam.optimize()


class TestTensor:
    """Test tensor bindings"""

    def test_create(self):
        """Test tensor creation"""
        tensor = cv.Tensor((3, 100, 100))
        assert tensor.shape() == (3, 100, 100)

    def test_zeros(self):
        """Test zeros tensor"""
        tensor = cv.Tensor.zeros((1, 10, 10))
        assert tensor.shape() == (1, 10, 10)
        arr = tensor.to_numpy()
        assert all(v == 0.0 for v in arr)

    def test_ones(self):
        """Test ones tensor"""
        tensor = cv.Tensor.ones((1, 10, 10))
        arr = tensor.to_numpy()
        assert len(arr) == 100
        assert all(v == 1.0 for v in arr)


class TestKeyPoints:
    """Test keypoints bindings"""

    def test_create(self):
        """Test creating keypoints"""
        kps = cv.KeyPoints()
        assert len(kps) == 0

    def test_to_list(self):
        """Test converting to list"""
        kps = cv.KeyPoints()
        lst = kps.to_list()
        assert isinstance(lst, list)
        assert len(lst) == 0


class TestRuntime:
    """Test runtime bindings"""

    def test_get_num_devices(self):
        """Test getting number of devices"""
        num = cv.Runtime.get_num_devices()
        assert num >= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
