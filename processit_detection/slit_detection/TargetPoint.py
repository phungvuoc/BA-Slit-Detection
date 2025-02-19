#!/usr/bin/env python3
import numpy as np

class TargetPoint:
    def __init__(self, index: int, position: np.ndarray, orientation: np.ndarray) -> None:
        self.index = index
        self.position = position
        self.orientation = orientation
