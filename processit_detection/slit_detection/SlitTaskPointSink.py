#!/usr/bin/env python3
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import rospy
from slit_detection.TargetPoint import TargetPoint
from processit_msgs.msg import GeometricFeature
from geometry_msgs.msg import Vector3

class SlitTaskPointSink:
    def __init__(self, task_point_pub: rospy.Publisher) -> None:
        self.target_point_pub = task_point_pub

    def send(self, target_points: "list[TargetPoint]") -> None:
        pub_rate = rospy.Rate(400)
        geometric_feature = GeometricFeature()
        for point in target_points:
            geometric_feature.point_index = point.index
            geometric_feature.point = Vector3(*point.position)
            geometric_feature.direction = Vector3(*point.orientation)
            self.target_point_pub.publish(geometric_feature)
            # Reduce publishing rate to avoid losing messages
            if len(target_points) > 1:
                pub_rate.sleep()
