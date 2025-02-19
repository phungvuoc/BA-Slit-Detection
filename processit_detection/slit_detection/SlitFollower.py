#!/usr/bin/env python3
import os
import sys
import time
from typing import List

import numpy as np

from slit_detection.TargetPoint import TargetPoint


sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import rospy
from pathlib import Path
from slit_detection.ScanLineData import SlitScanLineData
from slit_detection.DetectionMode import DetectionMode
from slit_detection.slit_collector.SlitCollector import SlitCollector
from slit_detection.slit_detector.BaseSlitDetector import BaseSlitDetector
from slit_detection.slit_detector.SearchingSlitDetector import SearchingSlitDetector
from slit_detection.slit_detector.FollowingSlitDetector import FollowingSlitDetector
from slit_detection.SlitTaskPointSink import SlitTaskPointSink
from slit_detection.DetectionOutput import DetectionResult

class SlitFollower:
    PREFIX = "[SlitFollower]:"

    def __init__(self, detector_type, config) -> None:
        if detector_type != "slit":
            raise ValueError(f"Detector type {detector_type} not supported")
        self.config = config
        self.task_point_sink = None
        self.detection_result = DetectionResult()
        self.collector = SlitCollector()
        self.detector = BaseSlitDetector()
        self.searching_detector = SearchingSlitDetector()
        self.following_detector = FollowingSlitDetector()

    def setTaskPointPublisher(self, task_point_publisher: rospy.Publisher):
        if self.task_point_sink is not None:
            return
        self.task_point_sink = SlitTaskPointSink(task_point_publisher)

    def setViewFieldLimits(self, field_upper_limit: float, field_lower_limit: float):
        self.collector.setViewFieldLimits(field_upper_limit, field_lower_limit)

    def updateMode(self, mode):
        if mode == DetectionMode.SEARCHING:
            self.collector.setNextWindowOffset(self.config.DIST_WELDSEARCH)
            self.detector = self.searching_detector

        elif mode == DetectionMode.PAUSING:
            pass

        elif mode == DetectionMode.FOLLOWING:
            self.collector.clearCollectedData()
            self.collector.setNextWindowOffset(
                0.003 # Check this with end_offset (DEFAULT_END_OFFSET = 0.002) to detect end of slit
            )  # TODO: pre-defined in config file
            self.detector = self.following_detector
            self.detector.updateResult(self.detection_result)

        elif mode == DetectionMode.ABORTED:
            self.__clear()

        else:
            raise ValueError(f"Mode {mode} not supported")

    def processScanLine(self, new_scan_line_data: SlitScanLineData) -> bool:
        # check if collector and detector are initialized
        if self.collector is None or self.detector is None:
            return False

        # collect scan line data
        success = self.collector.collect(new_scan_line_data)
        if not success:
            rospy.logerr(f"{self.PREFIX} Failed to collect scan line data")
            return False

        # check if window is reached
        reached = self.collector.isWindowReached()
        if not reached:
            return True

        # get window data
        scan_group = self.collector.getWindowData()
        if scan_group is None:
            return False
        self.collector.current_window_size = 0.0

        # detector processes to detect slit and task points
        detection_output = self.detector.detect(scan_group)
        if detection_output is None or self.task_point_sink is None:
            return False

        # publish task points
        if detection_output.success:
            new_target_points = detection_output.result.target_points
            self.task_point_sink.send(new_target_points)
            self.detection_result.main_plane = detection_output.result.main_plane
            self.detection_result.slit_medial_axis_line = (
                detection_output.result.slit_medial_axis_line
            )
            self.detection_result.slit = detection_output.result.slit
            self.detection_result.target_points.extend(new_target_points)
            self.detector.updateResult(self.detection_result)
            self.collector.scan_manager.clear()
            # For Evaluation
            rospy.logwarn(f"{self.PREFIX} continue_flag: {detection_output.continue_flag}")
            rospy.loginfo(
                f"{self.PREFIX} Total number of target points: {len(self.detection_result.target_points)}"
            )
            enable_save_target_points = False
            if enable_save_target_points and detection_output.result.slit.isEndDetected():
                to_save_target_points: List[TargetPoint] = self.detection_result.target_points
                to_save_position = np.array([point.position for point in to_save_target_points])
                rospy.loginfo(f"{self.PREFIX} Saving target {len(to_save_position)} positions")
                file_name = "slit_target_points_" + time.strftime("%Y%m%d-%H%M%S") + ".npy"
                log_dir: Path = Path(rospy.get_param("SCANDATA_DIR")) # type: ignore
                log_file = log_dir / file_name
                try:
                    log_file.parent.mkdir(exist_ok=True, parents=True)
                    rospy.logwarn(f"{self.PREFIX} Saving target points to {log_file}")
                    np.save(log_file, to_save_position)
                except Exception as e:
                    rospy.logerr(f"{self.PREFIX} Failed to save target points: {e}")
            return detection_output.continue_flag
        else:
            return False

    ##########################################
    def __clear(self):
        self.detection_result = DetectionResult()
        self.collector.reset()
        self.detector.reset()
        self.searching_detector.reset()
        self.following_detector.reset()


# %%
if __name__ == "__main__":
    pass
