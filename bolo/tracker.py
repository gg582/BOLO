"""Kalman Filter and Multi-Object Tracker for Overhead Head Tracking.

Implements constant velocity Kalman Box Tracker and bipartite matching
via Hungarian algorithm (scipy.optimize.linear_sum_assignment) for robust
person counting and false-positive suppression in CCTV streams.
"""

from typing import List, Tuple
import numpy as np
from scipy.optimize import linear_sum_assignment


class KalmanBoxTracker:
    """State space: [cx, cy, s, r, v_cx, v_cy, v_s]^T
    where:
    - cx, cy: center coordinates
    - s: scale (box area = w * h)
    - r: aspect ratio (w / h)
    - v_*: first derivatives (velocities)
    """

    count = 0

    def __init__(self, bbox: np.ndarray, score: float = 1.0):
        # State vector [7, 1]
        self.kf_dim_z = 4
        self.kf_dim_x = 7

        self.x = np.zeros((self.kf_dim_x, 1), dtype=np.float32)
        self.x[:4, 0] = self.convert_bbox_to_z(bbox)

        # State transition matrix F
        self.F = np.eye(self.kf_dim_x, dtype=np.float32)
        for i in range(3):
            self.F[i, i + 4] = 1.0

        # Measurement matrix H
        self.H = np.zeros((self.kf_dim_z, self.kf_dim_x), dtype=np.float32)
        self.H[:4, :4] = np.eye(self.kf_dim_z, dtype=np.float32)

        # Covariance matrix P
        self.P = np.eye(self.kf_dim_x, dtype=np.float32) * 10.0
        self.P[4:, 4:] *= 100.0  # high uncertainty on initial velocities

        # Process noise Q
        self.Q = np.eye(self.kf_dim_x, dtype=np.float32)
        self.Q[4:, 4:] *= 0.01

        # Measurement noise R
        self.R = np.eye(self.kf_dim_z, dtype=np.float32)
        self.R[2:, 2:] *= 10.0

        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = []
        self.hits = 1
        self.hit_streak = 1
        self.age = 0
        self.score = float(score)

    @staticmethod
    def convert_bbox_to_z(bbox: np.ndarray) -> np.ndarray:
        """[x1, y1, x2, y2] -> [cx, cy, s, r]"""
        w = max(bbox[2] - bbox[0], 1e-4)
        h = max(bbox[3] - bbox[1], 1e-4)
        cx = bbox[0] + w / 2.0
        cy = bbox[1] + h / 2.0
        s = w * h
        r = w / float(h)
        return np.array([cx, cy, s, r], dtype=np.float32)

    @staticmethod
    def convert_x_to_bbox(x: np.ndarray) -> np.ndarray:
        """[cx, cy, s, r] -> [x1, y1, x2, y2]"""
        s = max(x[2], 1e-4)
        r = max(x[3], 1e-4)
        w = np.sqrt(s * r)
        h = s / w
        cx, cy = x[0], x[1]
        return np.array([cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0], dtype=np.float32)

    def predict(self) -> np.ndarray:
        """Advance state vector via Kalman predict step."""
        if (self.x[6] + self.x[2]) <= 0:
            self.x[6] = 0.0
        self.x = np.dot(self.F, self.x)
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(self.convert_x_to_bbox(self.x[:4, 0]))
        return self.history[-1]

    def update(self, bbox: np.ndarray, score: float = 1.0):
        """Update state vector with new detection measurement."""
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        self.score = float(score)

        z = self.convert_bbox_to_z(bbox).reshape(-1, 1)
        # Innovation y = z - H * x
        y = z - np.dot(self.H, self.x)
        # Innovation covariance S = H * P * H^T + R
        S = np.dot(np.dot(self.H, self.P), self.H.T) + self.R
        # Kalman gain K = P * H^T * S^-1
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))

        # Updated state and covariance
        self.x = self.x + np.dot(K, y)
        self.P = self.P - np.dot(np.dot(K, self.H), self.P)

    def get_state(self) -> np.ndarray:
        """Returns current bounding box estimate [x1, y1, x2, y2]."""
        return self.convert_x_to_bbox(self.x[:4, 0])


def iou_batch(bb_test: np.ndarray, bb_gt: np.ndarray) -> np.ndarray:
    """Compute pairwise IoU matrix between detections and tracker predictions."""
    if len(bb_test) == 0 or len(bb_gt) == 0:
        return np.zeros((len(bb_test), len(bb_gt)), dtype=np.float32)

    bb_test = np.expand_dims(bb_test, 1)  # [N, 1, 4]
    bb_gt = np.expand_dims(bb_gt, 0)      # [1, M, 4]

    xx1 = np.maximum(bb_test[..., 0], bb_gt[..., 0])
    yy1 = np.maximum(bb_test[..., 1], bb_gt[..., 1])
    xx2 = np.minimum(bb_test[..., 2], bb_gt[..., 2])
    yy2 = np.minimum(bb_test[..., 3], bb_gt[..., 3])

    w = np.maximum(0.0, xx2 - xx1)
    h = np.maximum(0.0, yy2 - yy1)
    wh = w * h

    area_test = (bb_test[..., 2] - bb_test[..., 0]) * (bb_test[..., 3] - bb_test[..., 1])
    area_gt = (bb_gt[..., 2] - bb_gt[..., 0]) * (bb_gt[..., 3] - bb_gt[..., 1])

    union = area_test + area_gt - wh
    return wh / np.clip(union, 1e-6, None)


class AvianHeadTracker:
    """Overhead Head Multi-Object Tracker with Kalman Filtering & False-Positive Rejection.

    Filters:
    - min_hits: minimum consecutive detections required before confirming an object
                (eliminates transient 1-2 frame false-positives).
    - max_age: maximum frames to keep track alive during occlusions / missed detections.
    - iou_threshold: matching IoU threshold for Hungarian assignment.
    """

    def __init__(self, max_age: int = 15, min_hits: int = 3, iou_threshold: float = 0.3):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers: List[KalmanBoxTracker] = []
        self.frame_count = 0
        self.total_unique_count = 0
        self.counted_track_ids = set()

    def update(self, dets: np.ndarray) -> np.ndarray:
        """Update tracker with detections: [N, 5] (x1, y1, x2, y2, score).

        Returns:
            np.ndarray: Active confirmed tracks [M, 6] (x1, y1, x2, y2, track_id, score)
        """
        self.frame_count += 1

        # 1. Predict all current tracks with Kalman filter
        trks = np.zeros((len(self.trackers), 4), dtype=np.float32)
        to_del = []
        for t, trk in enumerate(self.trackers):
            pos = trk.predict()
            if np.any(np.isnan(pos)):
                to_del.append(t)
            else:
                trks[t, :] = pos

        for t in reversed(to_del):
            self.trackers.pop(t)

        # 2. Match detections with predicted tracks using Hungarian algorithm
        matched, unmatched_dets, unmatched_trks = self._associate_detections_to_trackers(dets, trks)

        # 3. Update matched trackers
        for m in matched:
            det_idx, trk_idx = m[0], m[1]
            self.trackers[trk_idx].update(dets[det_idx, :4], score=dets[det_idx, 4])

        # 4. Create new trackers for unmatched detections
        for i in unmatched_dets:
            trk = KalmanBoxTracker(dets[i, :4], score=dets[i, 4])
            self.trackers.append(trk)

        # 5. Extract confirmed active tracks and purge dead tracks
        ret = []
        i = len(self.trackers)
        for trk in reversed(self.trackers):
            i -= 1
            d = trk.get_state()
            # Object confirmation filter: must have hit streak >= min_hits to suppress false alarms
            if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits):
                ret.append(np.array([d[0], d[1], d[2], d[3], trk.id, trk.score], dtype=np.float32))
                if trk.id not in self.counted_track_ids:
                    self.counted_track_ids.add(trk.id)
                    self.total_unique_count += 1
            # Purge dead tracks
            if trk.time_since_update > self.max_age:
                self.trackers.pop(i)

        if len(ret) > 0:
            return np.stack(ret)
        return np.zeros((0, 6), dtype=np.float32)

    def _associate_detections_to_trackers(
        self, dets: np.ndarray, trks: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if len(trks) == 0:
            return np.empty((0, 2), dtype=int), np.arange(len(dets)), np.empty((0,), dtype=int)
        if len(dets) == 0:
            return np.empty((0, 2), dtype=int), np.empty((0,), dtype=int), np.arange(len(trks))

        iou_matrix = iou_batch(dets[:, :4], trks)

        if min(iou_matrix.shape) > 0:
            row_ind, col_ind = linear_sum_assignment(-iou_matrix)
            matched_indices = np.stack([row_ind, col_ind], axis=1)
        else:
            matched_indices = np.empty((0, 2), dtype=int)

        unmatched_dets = []
        for d, det in enumerate(dets):
            if d not in matched_indices[:, 0]:
                unmatched_dets.append(d)

        unmatched_trks = []
        for t, trk in enumerate(trks):
            if t not in matched_indices[:, 1]:
                unmatched_trks.append(t)

        matches = []
        for m in matched_indices:
            if iou_matrix[m[0], m[1]] < self.iou_threshold:
                unmatched_dets.append(m[0])
                unmatched_trks.append(m[1])
            else:
                matches.append(m.reshape(1, 2))

        if len(matches) == 0:
            matches = np.empty((0, 2), dtype=int)
        else:
            matches = np.concatenate(matches, axis=0)

        return matches, np.array(unmatched_dets), np.array(unmatched_trks)
