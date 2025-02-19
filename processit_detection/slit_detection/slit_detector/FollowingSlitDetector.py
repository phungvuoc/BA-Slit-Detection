#!/usr/bin/env python3
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import rospy
import numpy as np
import open3d as o3d
import slit_detector.SlitDetectorMath as math_utils
from slit_detector.BaseSlitDetector import BaseSlitDetector
from processit_core.line import Line
from processit_core.plane import Plane
from processit_core.pose_tuple import PoseTuple
from slit_detector.Slit import Slit
from slit_detection.DetectionOutput import DetectionResult, DetectionOutput
from slit_detection.ScanLineData import SlitScanLineData
from slit_detection.slit_detector.LineFitter import LineFitter
from slit_detection.TargetPoint import TargetPoint

class FollowingSlitDetector(BaseSlitDetector):
    PREFIX = "[FollowingSlitDetector]:"

    def __init__(self) -> None:
        super().__init__()

    def detect(self, scan_group: 'list[SlitScanLineData]'):
        start_time = rospy.Time.now()
        detection_output = DetectionOutput(
            result=DetectionResult(),
            success=False,
            continue_flag=False
        )

        # %% check length of scan_group
        if len(scan_group) < 2:
            rospy.logwarn(f"{self.PREFIX} Scan group too short to detect slit")
            return detection_output

        # get sensor poses of the first and last scan lines
        first_sensor_pose = scan_group[0].sensor_pose
        last_sensor_pose = scan_group[-1].sensor_pose
        if not self.__checkSensorPoses(first_sensor_pose, last_sensor_pose):
            return detection_output

        # %% Find the medial axis points of the scan group
        medial_axis_points = []
        end_line_points = []
        for scan_line in scan_group:
            # create point cloud
            scan_line_pcd = o3d.geometry.PointCloud()
            scan_line_pcd.points = o3d.utility.Vector3dVector(scan_line.scan_line_points)

            # pre-process
            scan_line_in_plane_pcd = self._preProcess(scan_line_pcd, self.main_plane)
            if scan_line_in_plane_pcd is None:
                return detection_output

            # process
            medial_axis_point = self._process(scan_line_in_plane_pcd, self.slit.getSlitWidth())
            if medial_axis_point is not None:
                medial_axis_points.append(medial_axis_point)
            else:
                end_line_points = scan_line_in_plane_pcd.points
                break

        # %% find target points
        target_points = self._findTargetPoints(medial_axis_points, end_line_points, first_sensor_pose, last_sensor_pose)
        # validate target points
        valid_target_points = self.__validateTargetPoints(target_points)

        # %% update detection output
        self.detection_result = DetectionResult(
            main_plane=self.main_plane,  # type: ignore
            slit_medial_axis_line=self.slit_medial_axis_line,  # type: ignore
            slit=self.slit,  # type: ignore
            target_points=valid_target_points,  # type: ignore
        )

        detection_output = DetectionOutput(
            result = self.detection_result,
            success =True,
            continue_flag = not self.slit.isEndDetected(),
        )
        # for evaluation
        end_time = rospy.Time.now()
        rospy.loginfo(f"{self.PREFIX} Detection time: {(end_time - start_time).to_sec():.2f} sec")
        points = np.vstack([scan_line.scan_line_points for scan_line in scan_group])
        rospy.logerr(f"{self.PREFIX} Number of scan lines: {len(scan_group)}")
        rospy.logerr(f"{self.PREFIX} Number of points: {len(points)}")
        return detection_output

    def updateResult(self, detection_result: DetectionResult):
        self.detection_result = detection_result
        self.main_plane: Plane = detection_result.main_plane
        self.slit_medial_axis_line: Line = detection_result.slit_medial_axis_line
        self.slit: Slit = detection_result.slit
        self.target_points: 'list[TargetPoint]' = detection_result.target_points

    ##########################################
    def __checkSensorPoses(self, first_sensor_pose: PoseTuple, last_sensor_pose: PoseTuple) -> bool:
        # Check if the y directions of the first and last sensor poses are in the same direction and parallel within tolerance
        is_y_dir_parallel = math_utils.checkTwoVectorsParallel(first_sensor_pose.y_dir, last_sensor_pose.y_dir, degree_tolerance=45)
        is_y_dir_same = np.dot(first_sensor_pose.y_dir, last_sensor_pose.y_dir) > 0
        if not is_y_dir_parallel or not is_y_dir_same:
            rospy.logerr(f"{self.PREFIX} Sensor poses' y directions are not aligned or parallel within tolerance")
            return False
        # %% compute the average y direction from the first and last sensor pose (green line)
        y_direction = (first_sensor_pose.y_dir + last_sensor_pose.y_dir) / 2
        normalized_y_direction = y_direction / np.linalg.norm(y_direction)
        dot_product = np.dot(normalized_y_direction, self.slit_medial_axis_line.direction)
        if abs(dot_product) < 0.25 or dot_product < 0:
            rospy.logerr(f"{self.PREFIX} Y direction of sensor pose is not in the same direction as the medial axis line")
            return False
        return True

    def _preProcess(self, pcd: o3d.geometry.PointCloud, main_plane: Plane):
        """
        ## Downsample point cloud
        ## Find all inliers from the point cloud to the main plane
        """
        # %% downsample point cloud
        pcd = pcd.voxel_down_sample(voxel_size=0.0005) # [m] TODO: find appropriate values

        # %% Find all inliers from the point cloud to the main plane
        inliers = []
        for point in pcd.points:
            distance = math_utils.distanceFromPointToPlane(point, main_plane)
            if distance < 0.0005:
                inliers.append(point)

        if len(inliers) < 3:
            rospy.logerr(f"{self.PREFIX} Not enough inliers to main plane")
            return None
        # TODO: update main plane model if needed

        # project inliers to the main plane
        inlier_points_projected = math_utils.projectPointsToPlane(np.array(inliers), main_plane)

        # %% create point cloud from inliers
        inliers_pcd = o3d.geometry.PointCloud()
        inliers_pcd.points = o3d.utility.Vector3dVector(inlier_points_projected)

        return inliers_pcd

    ##########################################
    def _process(
        self, scan_line_in_plane_pcd: o3d.geometry.PointCloud, slit_width: float
    ):
        """
        ## RANSAC line fitting and inlier indices
        ## distances between each pair of 2 adjacent points
        ## extract 2 slit boundary points (check distance with slit width)
        ## medial axis points as mean of slit boundary points
        """
        # %% RANSAC line fitting and inlier indices for the scan line
        points = np.asarray(scan_line_in_plane_pcd.points)
        line_model, indices = LineFitter.fitLineRansac(points, threshold=0.001)
        if line_model is None:
            rospy.logerr(f"{self.PREFIX} No line model found")
            return None

        # %% distances between each pair of 2 adjacent points
        inliers = np.asarray(scan_line_in_plane_pcd.points)[indices]
        inliers_with_distance = self.__sortInliersByDistanceToLinePoint(inliers, line_model)

        # %% extract 2 slit boundary points (check distance with slit width)
        slit_boundary_points = self.__findSlitBoundary(inliers_with_distance, slit_width)
        if len(slit_boundary_points) < 2:
            return None

        # %% medial axis point as mean of slit boundary points
        medial_axis_point = np.mean(slit_boundary_points, axis=0)

        return medial_axis_point

    def __sortInliersByDistanceToLinePoint(
        self, inliers: "list[np.ndarray]", line: Line
    ):
        if len(inliers) == 0:
            raise ValueError("There is no inlier")

        inliers_with_distance = []
        for inlier in inliers:
            vector = inlier - line.point
            dot_product = np.dot(vector, line.direction)
            distance_to_line_point = np.linalg.norm(vector)
            distance = (
                distance_to_line_point
                if dot_product > 0
                else (-1) * distance_to_line_point
            )
            inliers_with_distance.append((inlier, distance))

        inliers_with_distance.sort(key=lambda x: x[1])
        return inliers_with_distance

    def __findSlitBoundary(self, inliers_with_distance, slit_width: float):
        """
        """
        slit_boundary_points = []
        # %% if there is a gap in scan line
        for i in range(len(inliers_with_distance) - 1):
            inlier1 = inliers_with_distance[i][0]
            inlier2 = inliers_with_distance[i + 1][0]
            distance = inliers_with_distance[i + 1][1] - inliers_with_distance[i][1]
            if abs(distance - slit_width) < 0.001:
                slit_boundary_points.append(inlier1)
                slit_boundary_points.append(inlier2)
                break

        return slit_boundary_points

    ##########################################
    def _findTargetPoints(
        self,
        medial_axis_points: 'list[np.ndarray]',
        end_line_points: 'list[np.ndarray]',
        first_sensor_pose: PoseTuple,
        last_sensor_pose: PoseTuple) -> 'list[TargetPoint]':
        """
        ## refit slit medial axis line using SVD
        ## calculate target points
        """
        # %% Refit the slit medial axis line to the medial axis points
        slit_medial_axis_line = self.__refitSlitMedialAxisLine(
            medial_axis_points, self.slit_medial_axis_line
        )
        if slit_medial_axis_line is None:
            rospy.logerr(f"{self.PREFIX} Slit medial axis line not found")
            return []
        self.slit_medial_axis_line = slit_medial_axis_line

        # %% Check if the end width points can be detected
        radius = self.slit.getSlitWidth() / 2
        end_width_midpoint = self.__findEndWidthMidpoint(end_line_points, slit_medial_axis_line, radius)
        if end_width_midpoint is not None:
            rospy.logerr(f"{self.PREFIX} End width midpoint found")
            self.slit.setEndMidpoint(end_width_midpoint)
            start_width_midpoint = self.slit.getStartMidpoint()
            slit_length = np.linalg.norm(end_width_midpoint - start_width_midpoint)
            self.slit.setSlitLength(float(slit_length))
            rospy.loginfo(f"{self.PREFIX} Slit length: {slit_length * 1000:.2f} mm")

        # %% calculate target points
        # the first target position is the intersection point between the y plane of the first sensor pose and the medial axis line
        first_sensor_y_plane = self._getYPlaneOfSensorPose(first_sensor_pose)
        first_target_position = math_utils.intersectionPlaneAndLine(first_sensor_y_plane, slit_medial_axis_line)
        # check if the end width points are detected
        if self.slit.isEndDetected():
            # the last target position is the end width midpoint
            last_target_position = self.slit.getEndMidpoint() - slit_medial_axis_line.direction * self.DEFAULT_END_OFFSET # type: ignore
        else:
            # the last target position is the intersection point between the y plane of the last sensor pose and the medial axis line
            last_sensor_y_plane = self._getYPlaneOfSensorPose(last_sensor_pose)
            last_target_position = math_utils.intersectionPlaneAndLine(last_sensor_y_plane, slit_medial_axis_line)
        # check if the last to the first in the same direction as the medial axis line
        if np.dot(last_target_position - first_target_position, slit_medial_axis_line.direction) < 0:
            return []
        # orientation of the target points
        orientation = (-1) * self.main_plane.normal
        # target points
        target_points = self._calculateTargetPoints(first_target_position, last_target_position, orientation)

        return target_points

    def __refitSlitMedialAxisLine(self, medial_axis_points: 'list[np.ndarray]', slit_medial_axis_line: Line):
        last_direction = slit_medial_axis_line.direction
        last_target_points = self.target_points[-5:]
        # points to refit the line
        points = [target_point.position for target_point in last_target_points]
        points.extend(medial_axis_points)

        # Fit a line to the points using SVD
        line = LineFitter.fitLineSvd(np.array(points))
        if line is None:
            rospy.logerr(f"{self.PREFIX} Slit medial axis line not found")
            return slit_medial_axis_line

        dot_product = np.dot(last_direction, line.direction)
        if abs(dot_product) < 0.25:
            return None

        if dot_product < 0:
            line.direction = -line.direction

        return line

    def __findEndWidthMidpoint(self, end_line_points: 'list[np.ndarray]', slit_medial_axis_line: Line, radius: float):
        def findEndWidthPoints(end_line_points: 'list[np.ndarray]', slit_medial_axis_line: Line, radius: float):
            if len(end_line_points) < 2:
                return []

            end_width_points = []
            for query_point in end_line_points:
                distance_to_line = math_utils.distanceFromPointToLine(query_point, slit_medial_axis_line)
                # check if the distance to the line is less than the radius minus a small value
                if distance_to_line < radius - 0.0008: # [m] TODO: find appropriate values
                    end_width_points.append(query_point)
            return end_width_points

        # %% Main function
        width_plane_normal = -slit_medial_axis_line.direction
        # %% find the end width points
        end_width_points = findEndWidthPoints(end_line_points, slit_medial_axis_line, radius)
        if len(end_width_points) == 0:
            return None
        # %% find the end width midpoint as the intersection point between the width plane and the medial axis line
        end_width_plane = self._fitPlaneToPointsWithNormal(
            end_width_points, width_plane_normal
        )
        width_midpoint = math_utils.intersectionPlaneAndLine(plane=end_width_plane, line=slit_medial_axis_line)

        return width_midpoint

    ##########################################
    def __validateTargetPoints(self, new_target_points: 'list[TargetPoint]') -> 'list[TargetPoint]':
        # %% Main function
        if len(self.target_points) == 0:
            rospy.loginfo(f"{self.PREFIX} No previous target points")
            return []

        valid_target_points = []
        initial_target_point = self.target_points[0]
        last_target_point = self.target_points[-1]

        for new_target_point in new_target_points:
            angle_in_degree = math_utils.angleBetweenTwoVectors(new_target_point.orientation, last_target_point.orientation)
            distance_to_last = np.linalg.norm(new_target_point.position - last_target_point.position)
            distance_to_initial = np.linalg.norm(new_target_point.position - initial_target_point.position)
            distance_last_to_initial = np.linalg.norm(last_target_point.position - initial_target_point.position)

            is_angle_too_big = angle_in_degree > 15 # [degree] TODO: find appropriate values
            is_distance_too_far = distance_to_last > 0.05 # [m] TODO: find appropriate values
            is_distance_decreasing = distance_to_initial < distance_last_to_initial

            if is_angle_too_big or is_distance_too_far or is_distance_decreasing:
                continue
            valid_target_points.append(new_target_point)

        # re-define the indices of the target points
        lastest_target_index = self.target_points[-1].index
        for i, valid_target_point in enumerate(valid_target_points):
            valid_target_point.index = lastest_target_index + i + 1
        rospy.loginfo(f"{self.PREFIX} Found {len(valid_target_points)} valid target points")
        return valid_target_points


# %%
if __name__ == "__main__":
    pass
