#!/usr/bin/env python3
from typing import Tuple

import numpy as np

from processit_adapt.configs import config_base
from processit_adapt.path_generators.online_path_generator.online_path_generator import OnlinePathGenerator


class SlitPoseGenerator:

    def __init__(self, config_data: config_base.ConfigData) -> None:
        self.__config = config_data
        self.__path_generator = OnlinePathGenerator(self.__config)

    def getInitialTargetPose(self, current_position: np.ndarray, path_handler_data: Tuple[np.ndarray, np.ndarray]):
        kwargs = {
            "positions": path_handler_data[0],
            "orientations": path_handler_data[1],
            "pos_tcp": current_position,
            "init": True,
        }
        pose = self.__path_generator.calculatePoses(**kwargs)
        return pose

    def getTargetPose(
        self,
        current_position: np.ndarray,
        path_handler_data: Tuple[np.ndarray, np.ndarray],
        #dist_from_start: float,
        #dist_to_end: float
        ):
        kwargs = {
            "positions": path_handler_data[0],
            "orientations": path_handler_data[1],
            "pos_tcp": current_position,
            "init": False,
        }
        pose = self.__path_generator.calculatePoses(**kwargs)
        decel_dist, near_end = self.__path_generator.isApproachingEnd(
            self.__config.TERMINAL_END_DIST)

        return pose, not near_end, decel_dist

    def getTargetPoseLookAhead(
        self,
        current_position: np.ndarray,
        path_handler_data: Tuple[np.ndarray, np.ndarray],
        #dist_from_start: float,
        #dist_to_end: float
        ):
        kwargs = {
            "positions": path_handler_data[0],
            "orientations": path_handler_data[1],
            "pos_tcp": current_position,
            "init": False,
        }
        pose = self.__path_generator.calculatePosesLookAhead(**kwargs)
        decel_dist, near_end = self.__path_generator.isApproachingEnd(
            self.__config.TERMINAL_END_DIST)

        return pose, not near_end, decel_dist
