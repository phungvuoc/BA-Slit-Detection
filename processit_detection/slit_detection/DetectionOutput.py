#!/usr/bin/env python3
from dataclasses import dataclass
from dataclasses import field
from typing import List

from slit_detection.TargetPoint import TargetPoint
from slit_detector.Slit import Slit

from processit_core.line import Line
from processit_core.plane import Plane

@dataclass
class DetectionResult:
    main_plane: Plane = Plane()
    slit_medial_axis_line: Line = Line()
    slit: Slit = Slit()
    target_points: List[TargetPoint] = field(default_factory=list)


@dataclass
class DetectionOutput:
    result: DetectionResult = DetectionResult()
    success: bool = False
    continue_flag: bool = False
