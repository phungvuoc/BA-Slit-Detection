#!/usr/bin/env python3
from enum import Enum


class DetectionMode(Enum):
    ABORTED = 0 # Can be used as a default value
    SEARCHING = 1
    PAUSING = 2
    FOLLOWING = 3
