#!/usr/bin/env python3
from dataclasses import dataclass

import numpy as np

@dataclass
class MedialAxisPoint:
    center_point: np.ndarray
    radius: float
    query_point: np.ndarray
    query_normal: np.ndarray
    point_of_contact: np.ndarray
    separation_angle: float
