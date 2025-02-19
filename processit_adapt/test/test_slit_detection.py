#!/usr/bin/env python3
import getpass
import unittest

import numpy as np
import rospy
from scipy.spatial.transform import Rotation
from std_srvs.srv import Trigger
from std_srvs.srv import TriggerRequest

import processit_adapt.test_utils as utils
from processit_adapt import moveit_pipeline_interface
from processit_adapt.configs import config
from processit_core import ros_util
from processit_core.pose_tuple import PoseTuple
from processit_msgs.srv import InitAdaptRequest

global_options = config.GlobalOptions()
PARSING_KEY = "smart_seam_tracking"
TEST_CASE = 1 # 1, 2, 3; 0 to run all test cases

class TestSlitDetection(unittest.TestCase):
    PREFIX = "TestSlitDetection: "
    ##########################################
    # SETUP + UTILS
    ##########################################

    @classmethod
    def setUpClass(cls):
        cls.moveit_pipeline_interface = moveit_pipeline_interface.MoveitPipelineInterface()
        start_pose = PoseTuple(
            pos=np.array([0.75, 0.43, 0.77]),
            ori=Rotation.from_euler('xyz', [180, 30, 20], degrees=True).as_quat()
        )
        cls.start_pose = start_pose
        cls.controller_emulator = utils.UrControllerEmulator()

        cls.srv_start_slit_detection = ros_util.waitAndInitService(
            "/processit_detection/slit_detection/startSlitDetection", Trigger)
        cls.srv_end_slit_detection = ros_util.waitAndInitService(
            "/processit_detection/slit_detection/endSlitDetection", Trigger)

    @classmethod
    def setupDefaultInitAdaptRequest(cls, parsing_key: str) -> InitAdaptRequest:
        req = InitAdaptRequest()
        # Check if currently on docker/root (-> CI)
        rospy.logwarn(f"user: {getpass.getuser()}")

        req.uuid = "sim_individual"

        req.parsing_key = parsing_key
        req.lang_code = "en"  # currently selected language for error text lookups
        req.use_extended_input = True

        # extended_mode: welding
        if req.parsing_key == "seam_pilot":
            req.speed = 30  # [cm/min]
        elif req.parsing_key == "smart_seam_tracking":
            req.speed = 5  # [mm/s]

        req.push_pull_angle = 0
        req.welding_angle_offset = 0
        req.stickout = 10 + global_options.STICKOUT_OFFSET_CUSTOM  # [mm]
        req.x_offset = 0

        # extended_mode: drive path
        # "pecrit_distance", "pecrit_arc_length", "pecrit_pose"
        req.path_end_criterion = "pecrit_arc_length"
        req.pecrit_distance = 5000  # [mm]
        req.pecrit_arc_length = 100  # [mm]
        # req.pecrit_pose = "p[0.25313, -0.7742, -0.00681, -1.9065, 1.69442, 0.60580]"  # Pose string from robot X/Y/Z/RX/RY/RZ
        req.pecrit_pose = "p[0, 0, 0, 0, 0, 0]"  # Pose string from robot X/Y/Z/RX/RY/RZ
        req.stitch_seam = False
        req.stitch_seam_amount = 3.0
        req.stitch_seam_def_type = "stitch_seam_def_type_via_gap"  # "stitch_seam_def_type_via_gap", "stitch_seam_def_type_via_total_length"
        req.stitch_seam_total_length = 200  # [mm]
        req.stitch_seam_gap = 30  # [mm]

        req.seqcor_angle = "0"
        req.seqcor_offset = 100
        req.seqcor_height = 70
        req.panning_start = False  # corner_behaviour
        req.panning_end = False  # corner_behaviour

        # extended_mode: detection
        # Welding geometry via value of GeometryType: ['fillet_inside', 'fillet_outside', 'square_butt', 'toolpath_following', 'calibration', 'slit']
        req.geometry_type = "slit"
        req.search_field_upper_limit = 60.0  # [mm]
        req.search_field_lower_limit = 0  # [mm]
        req.weld_offline = False

        return req

    def moveToPose(self, target_pose: PoseTuple):
        rospy.loginfo(f"{self.PREFIX} Moving to pose")
        speed = 0.035  # [m/s]
        success = self.moveit_pipeline_interface.moveToTarget(target_pose, "PTP", speed)
        if not success:
            rospy.logerr("Failed to move to target pose")
            return success
        rospy.sleep(0.5)

        update_rate = rospy.Rate(10)
        n = 0
        while n < 200:
            current_pos = self.moveit_pipeline_interface.getPose().pos
            if np.linalg.norm(current_pos - target_pose.pos) < 0.01:
                return True
            update_rate.sleep()
            n += 1

        rospy.sleep(0.5)
        return False

    ##########################################
    # TESTS
    ##########################################
    def test_slit_1(self):
        if TEST_CASE != 0 and TEST_CASE != 1:
            return
        rospy.loginfo(f"{self.PREFIX} Test slit 1")
        start_pose = PoseTuple(
            pos=np.array([0.748934, 0.3459651, 0.78195901]), #0.34 - 29
            ori=np.array([0.9813458, 0.00961546, -0.1909663, 0.0199961]),
        )
        success = self.moveToPose(start_pose)
        rospy.sleep(10)
        if not success:
            return

        req = self.setupDefaultInitAdaptRequest(PARSING_KEY)
        success, message = self.controller_emulator.mainRunningLoop(req)
        rospy.sleep(5)

    def test_slit_2(self):
        if TEST_CASE != 0 and TEST_CASE != 2:
            return
        rospy.loginfo(f"{self.PREFIX} Test slit 2")
        start_pose = PoseTuple(
            pos=np.array([0.748934, 0.2699651, 0.78195901]),
            ori=np.array([0.9813458, 0.00961546, -0.1909663, 0.0199961]),
        )
        success = self.moveToPose(start_pose)
        rospy.sleep(10)
        if not success:
            return
        req = self.setupDefaultInitAdaptRequest(PARSING_KEY)
        success, message = self.controller_emulator.mainRunningLoop(req)
        rospy.sleep(5)

    def test_slit_3(self):
        if TEST_CASE != 0 and TEST_CASE != 3:
            return
        rospy.loginfo(f"{self.PREFIX} Test slit 3")
        start_pose = PoseTuple(
            pos=np.array([0.748934, 0.1899651, 0.78195901]),
            ori=np.array([0.9813458, 0.00961546, -0.1909663, 0.0199961]),
        )
        success = self.moveToPose(start_pose)
        rospy.sleep(10)
        if not success:
            return
        req = self.setupDefaultInitAdaptRequest(PARSING_KEY)
        success, message = self.controller_emulator.mainRunningLoop(req)
        rospy.sleep(5)

    def test_get_pose(self):
        return # Skip this test
        pose = self.moveit_pipeline_interface.getPose()
        rospy.loginfo(f"Current pose: {pose.pos} + {pose.ori}")

    def test_slit_recording(self):
        return # Skip this test
        start_pose = PoseTuple(
            pos=np.array([0.75, 0.37, 0.77]),
            ori=Rotation.from_euler('xyz', [180, 10, 20], degrees=True).as_quat()
        )
        end_pose = PoseTuple(
            pos=np.array([0.75, 0.32, 0.77]),
            ori=Rotation.from_euler('xyz', [180, 10, 20], degrees=True).as_quat()
        )

        self.moveToPose(start_pose)
        rospy.sleep(2)
        res = self.srv_start_slit_detection(TriggerRequest())
        if not res.success:
            rospy.logerr("Failed to start slit detection")
            return

        # rospy.sleep(1)

        self.moveToPose(end_pose)
        rospy.sleep(2)
        res = self.srv_end_slit_detection(TriggerRequest())
        if not res.success:
            rospy.logerr("Failed to end slit detection")
            return


# %%
if __name__ == "__main__":

    rospy.init_node("test_slit_detection")

    # Run all test cases
    import rostest
    rostest.rosrun("processit_adapt", "test_slit_detection", TestSlitDetection)
