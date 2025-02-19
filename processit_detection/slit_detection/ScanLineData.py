#!/usr/bin/env python3
from dataclasses import dataclass

import numpy as np
from geometry_msgs.msg import Transform

from processit_core.pose_tuple import PoseTuple


@dataclass
class SlitScanLineData:
    scan_line_points: np.ndarray
    sensor_to_world: Transform
    world_to_sensor: Transform
    sensor_pose: PoseTuple
