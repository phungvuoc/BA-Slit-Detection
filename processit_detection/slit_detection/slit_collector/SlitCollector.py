#!/usr/bin/env python3
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

import rospy
import numpy as np
from slit_detection.ScanLineData import SlitScanLineData
from slit_detection.slit_detector import SlitDetectorMath
from slit_detection.slit_collector.SlitScanManager import SlitScanManager


class SlitCollector:
    DEFAULT_SEARCH_FIELD_UPPER_LIMIT = 0.2 # [m]
    DEFAULT_SEARCH_FIELD_LOWER_LIMIT = 0.0 # [m]

    def __init__(self) -> None:
        self.scan_manager = SlitScanManager()
        self.window_offset = 0.0
        self.current_window_size = 0.0
        self.search_field_upper_limit = self.DEFAULT_SEARCH_FIELD_UPPER_LIMIT
        self.search_field_lower_limit =  self.DEFAULT_SEARCH_FIELD_LOWER_LIMIT

    def setViewFieldLimits(self, field_upper_limit: float, field_lower_limit: float):
        if field_upper_limit < 0 or field_lower_limit < 0:
            rospy.logerr("Upper and lower limits for the search field must be positive")
            return
        self.search_field_upper_limit = field_upper_limit
        self.search_field_lower_limit = field_lower_limit

    def collect(self, new_scan_line_data: SlitScanLineData) -> bool:
        # check number of scan line points
        if len(new_scan_line_data.scan_line_points) < 100:
            rospy.logwarn("Not enough points in scan line data")
            return False
        # check the distance to the previous scan line
        if not self.scan_manager.checkEmpty():
            # check distance to previous scan line
            distance_to_previous = self.__distanceToPreviousScan(new_scan_line_data)
            # Skip if the distance is too small
            if distance_to_previous < 1e-4: # [m] TODO: find appropriate values
                return True
            # Error when scan line data is not received continuously
            if distance_to_previous > 8e-3: # [m] TODO: find appropriate values
                rospy.logerr(f"Distance to previous scan line is too large: {distance_to_previous}")
                return False
            # update window size
            self.current_window_size += distance_to_previous

        # crop scan line data to view field limits
        filterd_scan_line = self.__viewFieldFilter(new_scan_line_data)
        # add scan line data to scan manager
        self.scan_manager.addScan(filterd_scan_line)
        return True

    def setNextWindowOffset(self, offset: float):
        if np.isclose(offset, 0.0, atol=1e-9):
            rospy.logerr("Window offset must be greater than zero")
            return
        self.window_offset = offset

    def isWindowReached(self) -> bool:
        if self.window_offset == 0.0 or self.current_window_size == 0.0:
            return False
        diff = self.current_window_size - self.window_offset
        reached = diff > -1.1e-4 # [m] TODO: find appropriate values
        return bool(reached)

    def getWindowData(self):
        scan_lines = self.scan_manager.getAllScanLines()
        return scan_lines

    def resetWindow(self):
        self.current_window_size = 0.0
        # self.scan_manager.clear()
        # self.window_offset = 0.0

    def reset(self):
        self.scan_manager.clear()
        self.window_offset = 0.0
        self.current_window_size = 0.0
        self.search_field_upper_limit = self.DEFAULT_SEARCH_FIELD_UPPER_LIMIT
        self.search_field_lower_limit = self.DEFAULT_SEARCH_FIELD_LOWER_LIMIT

    def clearCollectedData(self):
        self.scan_manager.clear()
        self.window_offset = 0.0
        self.current_window_size = 0.0

    ##########################################
    def __distanceToPreviousScan(self, new_scan_line: SlitScanLineData):
        previous_scan_line = self.scan_manager.getLastScanLine()
        if previous_scan_line is None:
            raise ValueError("No previous scan line data")

        previous_sensor_pose = previous_scan_line.sensor_pose.pos
        new_sensor_pose = new_scan_line.sensor_pose.pos
        distance_between_two_sensor_poses = np.linalg.norm(previous_sensor_pose - new_sensor_pose)
        return distance_between_two_sensor_poses

    def __viewFieldFilter(self, scan_line_data: SlitScanLineData):
        """
        Filters the scan line data by the view field limits to get ROI (Region of Interest).
        This method filters out the points that have the x coordinate in the sensor coordinate system
        outside the view field limits. The sensor only takes the points along the x-axis (red line).
        Args:
            scan_line_data (SlitScanLineData): The scan line data containing points and transformation
                                               information.
        Returns:
            SlitScanLineData: The filtered scan line data with points within the view field limits.
        """
        if self.search_field_upper_limit < 0 or self.search_field_lower_limit < 0:
            rospy.logerr("Upper and lower limits for the search field must be positive")
            return scan_line_data

        # Get the view field limits
        max_view = self.search_field_upper_limit / 2 # [mm] To get the points in both sides from the center
        min_view = self.search_field_lower_limit / 2 # [mm] To remove the points that are too close to the center
        # Convert scan line points to a numpy array
        points_in_world_frame = np.array(scan_line_data.scan_line_points)
        # Transform points from world frame to sensor frame
        points_in_sensor_frame = SlitDetectorMath.transformPointsByTransform(
            points_in_world_frame, scan_line_data.world_to_sensor)
        # Create a boolean mask for points within the view field limits
        valid_points = np.logical_and(
            np.abs(points_in_sensor_frame[:, 0]) > min_view,  # Points greater than min_view
            np.abs(points_in_sensor_frame[:, 0]) < max_view   # Points less than max_view
        )
        # Filter the scan line points to only include valid points
        scan_line_data.scan_line_points = points_in_world_frame[valid_points]
        # Return the filtered scan line data
        return scan_line_data


# %%
if __name__ == "__main__":
    pass
