#!/usr/bin/env python3
import os
import sys
from collections import defaultdict
from typing import Dict
from typing import List

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

import numpy as np
import slit_detector.SlitDetectorMath as math_utils
from processit_core.line import Line
from processit_core.plane import Plane
from slit_detector.Slit import Slit
from slit_detection.ScanLineData import SlitScanLineData
from slit_detection.DetectionOutput import DetectionResult
from slit_detection.TargetPoint import TargetPoint
from processit_core.pose_tuple import PoseTuple

class BaseSlitDetector:
    DEFAULT_START_OFFSET = 0.002 # [m] TODO: pre-defined in config file or get from process parameters
    DEFAULT_END_OFFSET = 0.002 # [m] TODO: pre-defined in config file or get from process parameters

    def __init__(self) -> None:
        self.detection_result = DetectionResult()
        self.main_plane = Plane()
        self.slit_medial_axis_line = Line()
        self.slit = Slit()
        self.target_points = []

    def detect(self, scan_group: 'list[SlitScanLineData]'):
        pass

    def reset(self):
        self.detection_result = DetectionResult()
        self.main_plane = Plane()
        self.slit_medial_axis_line = Line()
        self.slit = Slit()
        self.target_points = []

    def updateResult(self, detection_result: DetectionResult):
        pass

    ##########################################

    ##########################################
    def _calculateTargetPoints(self, start_position: np.ndarray, last_position: np.ndarray, orientation: np.ndarray):
        """
        """
        target_points = []
        # calculate the number of interpolation points
        distance = np.linalg.norm(last_position - start_position)
        number_interpolation_points = int(distance / 0.001) # [m] TODO: find appropriate values
        # interpolate the target points between the initial and the middle target points
        for i in range(0, number_interpolation_points):
            interpolation_factor = float(i / number_interpolation_points)
            target_position = math_utils.interpolatePoint(start_position, last_position, interpolation_factor)
            target_points.append(TargetPoint(index=i, position=target_position, orientation=orientation))
        return target_points

    def _getYPlaneOfSensorPose(self, sensor_pose: PoseTuple):
        a, b, c = sensor_pose.y_dir
        d = -np.dot(sensor_pose.y_dir, sensor_pose.pos)
        return Plane(np.array([a, b, c, d]))

    def _fitPlaneToPointsWithNormal(
        self, points: "list[np.ndarray]", normal: np.ndarray
    ):
        tolerance = 0.0001 # [m] Distance tolerance for considering a point to be on the plane, TODO: find appropriate values
        # Group points based on their projection on the normal vector
        projections = np.dot(points, normal)
        projection_groups: Dict[float, List[int]] = defaultdict(list)
        for i, proj in enumerate(projections):
            rounded_proj = round(proj / tolerance) * tolerance
            projection_groups[rounded_proj].append(i)
        # Find the projection value that has the most points
        max_points = 0
        optimal_d = 0
        for proj_value, point_indices in projection_groups.items():
            num_points = len(point_indices)
            if num_points > max_points:
                max_points = num_points
                optimal_d = -proj_value  # The d parameter in plane equation
        plane_model = np.array([normal[0], normal[1], normal[2], optimal_d])
        plane = Plane(plane_model)
        return plane


# %%
if __name__ == "__main__":
    pass
