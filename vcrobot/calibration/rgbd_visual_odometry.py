"""Metric two-frame RGB-D visual odometry for an Intel RealSense stream.

The reference depth image turns ORB keypoints into metric 3-D points.  PnP
then estimates their pose in the current color frame.  No IMU is required.
Transforms follow the project convention: ``T_XY`` maps frame Y into frame X.
"""

from dataclasses import dataclass
import math

import cv2
import numpy as np


def _transform(rotation, translation):
    result = np.eye(4, dtype=float)
    result[:3, :3] = np.asarray(rotation, dtype=float).reshape(3, 3)
    result[:3, 3] = np.asarray(translation, dtype=float).reshape(3)
    return result


def invert_transform(transform):
    transform = np.asarray(transform, dtype=float).reshape(4, 4)
    result = np.eye(4, dtype=float)
    result[:3, :3] = transform[:3, :3].T
    result[:3, 3] = -result[:3, :3].dot(transform[:3, 3])
    return result


def rotation_angle(rotation):
    cosine = (float(np.trace(rotation)) - 1.0) * 0.5
    return math.acos(max(-1.0, min(1.0, cosine)))


@dataclass(frozen=True)
class VisualOdometryResult:
    T_C1_C0: np.ndarray
    T_C0_C1: np.ndarray
    feature_matches: int
    depth_matches: int
    inliers: int
    reprojection_rmse_px: float

    def to_dict(self):
        return {
            "T_C1_C0": self.T_C1_C0.tolist(),
            "T_C0_C1": self.T_C0_C1.tolist(),
            "feature_matches": self.feature_matches,
            "depth_matches": self.depth_matches,
            "inliers": self.inliers,
            "inlier_ratio": self.inliers / max(1, self.depth_matches),
            "reprojection_rmse_px": self.reprojection_rmse_px,
            "translation_m": self.T_C0_C1[:3, 3].tolist(),
            "distance_m": float(np.linalg.norm(self.T_C0_C1[:3, 3])),
            "rotation_rad": rotation_angle(self.T_C0_C1[:3, :3]),
        }


def estimate_pose_from_correspondences(object_points, image_points,
                                       camera_matrix, distortion=None,
                                       reprojection_error_px=2.5,
                                       confidence=0.999, iterations=200):
    """Estimate ``T_C1_C0`` from reference 3-D/current 2-D correspondences."""
    object_points = np.asarray(object_points, dtype=np.float32).reshape(-1, 3)
    image_points = np.asarray(image_points, dtype=np.float32).reshape(-1, 2)
    camera_matrix = np.asarray(camera_matrix, dtype=float).reshape(3, 3)
    distortion = (np.zeros((5, 1), dtype=float) if distortion is None else
                  np.asarray(distortion, dtype=float).reshape(-1, 1))
    if len(object_points) != len(image_points) or len(object_points) < 6:
        raise ValueError("at least six matched RGB-D correspondences are required")
    success, rvec, tvec, inlier_indices = cv2.solvePnPRansac(
        object_points, image_points, camera_matrix, distortion,
        iterationsCount=int(iterations),
        reprojectionError=float(reprojection_error_px),
        confidence=float(confidence), flags=cv2.SOLVEPNP_EPNP)
    if not success or inlier_indices is None or len(inlier_indices) < 6:
        raise RuntimeError("RGB-D PnP failed or produced fewer than six inliers")
    indices = inlier_indices.reshape(-1)
    success, rvec, tvec = cv2.solvePnP(
        object_points[indices], image_points[indices], camera_matrix,
        distortion, rvec, tvec, True, flags=cv2.SOLVEPNP_ITERATIVE)
    if not success:
        raise RuntimeError("RGB-D PnP refinement failed")
    projected, _ = cv2.projectPoints(
        object_points[indices], rvec, tvec, camera_matrix, distortion)
    residual = projected.reshape(-1, 2) - image_points[indices]
    rmse = float(np.sqrt(np.mean(np.sum(residual * residual, axis=1))))
    rotation, _ = cv2.Rodrigues(rvec)
    return _transform(rotation, tvec), indices, rmse


class RGBDVisualOdometry:
    """Two-frame ORB + reference-depth + PnP estimator."""

    def __init__(self, camera_matrix, distortion=None, max_features=2500,
                 ratio_test=0.75, min_depth_m=0.15, max_depth_m=6.0,
                 min_inliers=20, max_rmse_px=3.0):
        self.camera_matrix = np.asarray(camera_matrix, dtype=float).reshape(3, 3)
        self.distortion = (np.zeros(5, dtype=float) if distortion is None else
                           np.asarray(distortion, dtype=float).reshape(-1))
        self.orb = cv2.ORB_create(nfeatures=int(max_features), fastThreshold=12)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        self.ratio_test = float(ratio_test)
        self.min_depth_m = float(min_depth_m)
        self.max_depth_m = float(max_depth_m)
        self.min_inliers = int(min_inliers)
        self.max_rmse_px = float(max_rmse_px)
        self.reference = None

    @staticmethod
    def _gray(image):
        image = np.asarray(image)
        if image.ndim == 2:
            return image
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    def set_reference(self, color_bgr, depth_m):
        gray = self._gray(color_bgr)
        depth = np.asarray(depth_m, dtype=np.float32)
        if depth.shape != gray.shape:
            raise ValueError("aligned depth and color must have identical dimensions")
        keypoints, descriptors = self.orb.detectAndCompute(gray, None)
        if descriptors is None or len(keypoints) < self.min_inliers:
            raise RuntimeError("reference frame has too few ORB features")
        self.reference = (keypoints, descriptors, depth.copy())
        return len(keypoints)

    def estimate(self, color_bgr):
        if self.reference is None:
            raise RuntimeError("set_reference must be called first")
        current_keypoints, current_descriptors = self.orb.detectAndCompute(
            self._gray(color_bgr), None)
        if current_descriptors is None:
            raise RuntimeError("current frame has no ORB descriptors")
        reference_keypoints, reference_descriptors, reference_depth = self.reference
        pairs = self.matcher.knnMatch(reference_descriptors, current_descriptors, k=2)
        matches = [pair[0] for pair in pairs if len(pair) == 2 and
                   pair[0].distance < self.ratio_test * pair[1].distance]
        fx, fy = self.camera_matrix[0, 0], self.camera_matrix[1, 1]
        cx, cy = self.camera_matrix[0, 2], self.camera_matrix[1, 2]
        object_points, image_points = [], []
        for match in matches:
            u, v = reference_keypoints[match.queryIdx].pt
            ui, vi = int(round(u)), int(round(v))
            if not (0 <= vi < reference_depth.shape[0] and
                    0 <= ui < reference_depth.shape[1]):
                continue
            depth = float(reference_depth[vi, ui])
            if not self.min_depth_m <= depth <= self.max_depth_m:
                continue
            object_points.append(((u - cx) * depth / fx,
                                  (v - cy) * depth / fy, depth))
            image_points.append(current_keypoints[match.trainIdx].pt)
        T_C1_C0, inliers, rmse = estimate_pose_from_correspondences(
            object_points, image_points, self.camera_matrix, self.distortion)
        if len(inliers) < self.min_inliers:
            raise RuntimeError(
                f"visual odometry rejected: {len(inliers)} < {self.min_inliers} inliers")
        if rmse > self.max_rmse_px:
            raise RuntimeError(
                f"visual odometry rejected: reprojection RMSE {rmse:.2f} px")
        return VisualOdometryResult(
            T_C1_C0, invert_transform(T_C1_C0), len(matches),
            len(object_points), len(inliers), rmse)
