#!/usr/bin/env python3
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import rospy
import numpy as np
from typing import Dict
from typing import Tuple
from std_msgs.msg import Int8
from std_srvs.srv import Empty
from std_srvs.srv import Trigger
from std_srvs.srv import TriggerRequest
from geometry_msgs.msg import TwistStamped
from processit_core import py_util, ros_util
from processit_core.pose_tuple import PoseTuple
from processit_msgs.srv import ReadDetectionAbortReason
from processit_msgs.srv import StartDetection
from processit_msgs.srv import StartDetectionRequest
from processit_adapt import weld_traj_controller
from processit_adapt.configs import config
from processit_adapt.adapt_status import STATUS
from processit_adapt.task_appliers import task_applier_base
from processit_adapt.configs.config_base import GeometryType
from processit_adapt.configs.process_parameters import ProcessParameters
from processit_adapt.task_appliers.task_slit_following.slit_pose_generator import SlitPoseGenerator
from processit_adapt.task_appliers.task_online_following.util import \
    end_evaluator

global_options = config.GlobalOptions()
MOTION = config.MOTION_MODES


class TaskSlitFollower(task_applier_base.TaskApplier):
    PREFIX = "[TaskSlitFollower]:"

    def __init__(self, *args, **kwargs):
        self.__initPublishers()
        super().__init__(*args, **kwargs)
        self.__initSubscribers()
        self.__initServices()

        self.processParameters = ProcessParameters()
        self.pose_generator = SlitPoseGenerator(self._config)
        self.__start_pose = PoseTuple()

    def __initPublishers(self):
        super()._initPublishers()
        self.__pub_twist = rospy.Publisher("/processit_adapt/robot_controller_bridge/servoTwist",
                                           TwistStamped,
                                           queue_size=1)

    def __initSubscribers(self):
        super()._initSubscribers()
        self.__sub_servo_status = rospy.Subscriber(
            "/moveit_servo_interface/status", Int8, self.__cb_servo_status)

    def __initServices(self):
        super()._initServices()
        # Detection services
        self._srv_startDetection = ros_util.waitAndInitService(
            "processit_detection/slit_detection/startSlitDetection", StartDetection)
        self._srv_endDetection = ros_util.waitAndInitService(
            "processit_detection/slit_detection/endSlitDetection", Trigger)
        self._srv_pauseDetection = ros_util.waitAndInitService(
            "processit_detection/slit_detection/pauseSlitDetection", Trigger)
        self._srv_resumeDetection = ros_util.waitAndInitService(
            "processit_detection/slit_detection/resumeSlitDetection", Trigger)
        self._srv_readAbortReason = ros_util.waitAndInitService(
            "processit_detection/slit_detection/abortSlitReason", ReadDetectionAbortReason)

        # For resolving errors
        self.__srv_resetServo = ros_util.waitAndInitService(
            "moveit_servo_interface/reset_servo_status", Empty)
        # For consecutive motions in simulation
        self.__srv_restartServo = ros_util.waitAndInitService(
            "/processit_adapt/moveit_servo_interface/restart", Empty)
    ######################### Callbacks #########################
    def __cb_servo_status(self, msg: Int8):
        """Gets servo status from MoveIt, save in variable
        variable used to stop realtime motion if anything strange happens during servoing"""
        self.__servo_status = msg.data

    ######################### Interface / Main Functions #########################
    def taskInitialize(self, param: ProcessParameters) -> Tuple[bool, STATUS]:
        #%% Check if geometry is correct
        if param.geometry != GeometryType.SLIT.value:
            rospy.logerr(f"{self.PREFIX} Wrong geometry type")
            return False, STATUS.ERROR_INITIALIZE
        rospy.logwarn(f"{self.PREFIX} taskInitialize")
        #%% Set process parameters
        self.processParameters = param
        self._setNominalStickout(0, 0, param.stickout, 0, param.welding_angle_offset, 0)
        #param.printProcessParameters()

        #%% Set parameters
        self.__start_pose = self._getStickoutInWorld()[1]

        success = True
        status = STATUS.NONE
        
        # for evaluation
        self.start_time = rospy.Time.now()
        return success, status

    def taskApproach(self) -> Tuple[bool, STATUS, bool]:
        """
        """
        success = False
        status = STATUS.NONE
        continue_flag = False

        rospy.logwarn(f"{self.PREFIX} taskApproach")
        self._activateMoveItController()

        # Do approach
        success = self.__approachSlit()
        if not success:
            self.__moveBackToStart()
            # TODO: read abort reason
            #     response:ReadDetectionAbortReasonResponse = self._srv_readAbortReason()
            #     code:STATUS = self.__translateAbortCode(int(response.response_integer))
            self._srv_endDetection(TriggerRequest())
            status = STATUS.ERROR_APPROACH
            continue_flag = False
        else:
            status = STATUS.NONE
            continue_flag = True

        # Reset servo status in case there were any errors
        self._activateRobotController()
        self.__srv_resetServo()
        if self.processParameters.identification == "sim_individual":
            rospy.sleep(5) # There is always delay in simulation to move or update the sensor
        return success, status, continue_flag

    def taskMain(self) -> Tuple[bool, STATUS, bool]:
        """
        """
        success = False
        status = STATUS.NONE

        rospy.logwarn(f"{self.PREFIX} taskMain")
        self._srv_resumeDetection(TriggerRequest())
        self._activateServoController()
        self.__srv_restartServo()

        # TODO get parameters and speed from process parameters
        speed = 0.002 # [m/s]
        # Default: Uses the accumulated arc length (the sum of distances between consecutive positions) as the end criteria.
        end_eval_param = ("end_arc_length", 0.5) # [m] Max arc length ~ max slit length - TODO: read from config
        servo_valid, dist_to_end, driven_length = self.__followSlit(speed, end_evaluator.init(*end_eval_param))
        self._updateStats(process_length=driven_length)
        if not servo_valid:
            self._srv_endDetection(TriggerRequest())
            success = False
            status = STATUS.ERROR_MAIN
        else:
            self._srv_endDetection(TriggerRequest())
            success = True
            status = STATUS.NONE

        self._activateRobotController()
        self.__srv_resetServo()

        return success, status, False # Continue flag is always False (no interval task)

    def taskExit(self) -> Tuple[bool, Dict[str, PoseTuple]]:
        rospy.logwarn(f"{self.PREFIX} taskExit")
        resp = {'pose': self._getStickoutInWorld()[1]}
        # End everything movement based and detection
        self._activateRobotController()
        self._srv_endDetection(TriggerRequest())
        self.__sub_servo_status.unregister()
        super()._shutdown()
        
        self.end_time = rospy.Time.now()
        duration = (self.end_time - self.start_time).to_sec()
        rospy.loginfo(f"{self.PREFIX} Total duration of process: {duration} seconds")
        return True, resp

    ######################### Task Approach #########################
    def __approachSlit(self) -> bool:
        # Start slit detection
        request = self.__getStartDetectionRequest()
        response = self._srv_startDetection(request)
        if not response.success:
            return False

        # Move to scan
        if not self.__moveScanning():
            return False

        # Wait for detection
        if not self.__waitForSearchingDetection():
            rospy.logerr(f"{self.PREFIX} No detection data")
            return False

        self._srv_pauseDetection(TriggerRequest())

        # Move approach
        if not self.__moveApproach():
            return False

        return True

    ######################### Helper Functions for Task Approach #########################
    def __moveBackToStart(self) -> bool:
        """Move back to starting pose
        """
        rospy.loginfo(f"{self.PREFIX} Moving to start pose")
        start_pose = self.__start_pose
        self._moveit_pipeline_interface.setStickoutTransform(
            self._T_tcp_to_stickout)
        success = self._moveit_pipeline_interface.moveCartesianTuples(
            [start_pose], self._config.V_APPROACH)
        self.__waitMoving(start_pose)
        return success

    def __getStartDetectionRequest(self) -> StartDetectionRequest:
        req = StartDetectionRequest()
        req.geometry = GeometryType(self.processParameters.geometry).value
        req.identification = self.processParameters.identification
        # Sensor view field
        req.search_field_upper_limit = self.processParameters.search_field_upper_limit
        req.search_field_lower_limit = self.processParameters.search_field_lower_limit
        return req

    def __moveScanning(self) -> bool:
        rospy.loginfo(f"{self.PREFIX} Start scanning")
        # Move to scan to detect slit and initial point
        self._moveit_pipeline_interface.setStickoutTransform(self._T_tcp_to_stickout)

        target = PoseTuple(np.array([0, self._config.DIST_WELDSEARCH, 0]))
        success = self._moveit_pipeline_interface.moveRelative(target, v=self._config.V_WELDSEARCH)

        # Wait
        rospy.loginfo(f"{self.PREFIX} Waiting for transferring scan data")
        self.__waitMoving(target)
        if self.processParameters.identification == "sim_individual":
            rospy.sleep(2) # There is always delay in simulation to move or update the sensor
        return success

    def __waitForSearchingDetection(self) -> bool:
        rospy.loginfo(f"{self.PREFIX} Waiting for detection")
        wait_rate = rospy.Rate(50)
        n = 0
        while n < 1000:
            # check path data more than 5 points
            if len(self._path_data_handler.positions) > 5:
                return True
            wait_rate.sleep()
            n += 1
        return False

    def __moveApproach(self) -> bool:
        rospy.loginfo(f"{self.PREFIX} Moving to approach")
        self._moveit_pipeline_interface.setStickoutTransform(self._T_tcp_to_stickout)
        current_pos = self._getStickoutInWorld()[1].pos

        initial_pose = self.pose_generator.getInitialTargetPose(current_pos, self._path_data_handler.getData())
        self._moveit_pipeline_interface.setStickoutTransform(self._T_tcp_to_stickout)
        success = self._moveit_pipeline_interface.moveApproach(initial_pose)
        self.__waitMoving(initial_pose)
        return success

    def __waitMoving(self, target_pose: PoseTuple):
        target_position = target_pose.pos
        wait_rate = rospy.Rate(50)
        n = 0
        while n < 200:
            current_position = self._getStickoutInWorld()[1].pos
            distance_to_target = np.linalg.norm(np.array(current_position) - np.array(target_position))
            if distance_to_target < 0.01:
                return
            wait_rate.sleep()
            n += 1
        rospy.sleep(0.5)

    ######################### Task Main #########################
    def __followSlit(self, speed: float, end_eval: end_evaluator.EndEvaluator) -> Tuple[bool, float, float]:
        """
        """
        rospy.logwarn(f"{self.PREFIX} Start following")
        rospy.loginfo(f"{self.PREFIX} speed: {speed * 1000} mm/s.")

        #%% Start End Evaluator
        pos_start = self._getStickoutInWorld()[1].pos
        end_eval.start(pos_start)
        dist_to_end_criteria = end_eval.evaluate(pos_start)[1]
        rospy.loginfo(f"{self.PREFIX} Distance to End Criteria (Max length): {dist_to_end_criteria * 1000} mm")

        #%% Initialize Controller
        if self._config.CONTROLLER_CHOICE == 1:
            rospy.loginfo(f"{self.PREFIX} followSlit: PID-Controller is active")
            realtime_controller = weld_traj_controller.WeldTrajectoryPIDController(self._config, self._sim, speed)
        if self._config.CONTROLLER_CHOICE == 2:
            rospy.loginfo(f"{self.PREFIX} followSlit: 2DoF-Controller is active")
            realtime_controller = weld_traj_controller.WeldTrajectoryPIDLookAhead(self._config, self._sim, speed)

        #%% Calculate deceleration_distance
        decel_dist_physics = 0.5 * speed**2 / (self._config.DECELERATION)
        decel_dist_control_loop = 1 / self._config.UPDATE_RATE * speed
        # Add 1 control loop distance to compensate for delay
        deceleration_distance = decel_dist_physics + decel_dist_control_loop
        rospy.loginfo(f"{self.PREFIX} Deceleration Distance: {deceleration_distance * 1000} mm")

        #%% Initialize variables for main loop
        n = 0 # Iteration counter
        trigger_deceleration = False # Deceleration trigger
        twist_tcp = [np.array([0, 0, 0]), np.array([0, 0, 0])] # Twist in TCP frame to move
        update_rate = rospy.Rate(self._config.UPDATE_RATE)

        #%% Main loop
        while self.__continueMotion(n):
            success, dist_to_end = self.__doMotion(end_eval, deceleration_distance, trigger_deceleration, realtime_controller, twist_tcp)
            if not success:
                break

            n += 1
            # Sleep at the end of each loop to make sure the sensor data is transferred
            # There is always delay in simulation to move or update the sensor
            if self.processParameters.identification == "sim_individual":
                if dist_to_end < 0.015: # [m]
                    rospy.sleep(speed * 100) # find appropriate value
                elif dist_to_end < 0.025: # [m]
                    rospy.sleep(speed * 50) # find appropriate value
                else:
                    rospy.sleep(speed * 25) # find appropriate value
            else:
                # to make sure the online dection has enough time to process the data
                # the distance to end is needed to keep a distance from 15 mm after experiment
                # but this is not suitable for the welding process when TCP always has to be hold after each moving
                # optimization is needed
                if dist_to_end < 0.1: # [m]
                    rospy.sleep(speed * 100) # find appropriate value
                else:
                    update_rate.sleep()

        #%% Send empty servo message to stop movement
        self.__sendTwist()
        rospy.loginfo(f"{self.PREFIX} End Reached: Seam Arc Length: {(end_eval.length * 1000):.2f} mm")
        rospy.loginfo(f"{self.PREFIX} End following")
        return self.__checkServoState(n), dist_to_end_criteria, end_eval.length

    ######################### Helper Functions for Task Main #########################
    def __doMotion(self, end_eval, deceleration_distance,  trigger_deceleration, realtime_controller, twist_tcp) -> Tuple[bool, float]:
        #%% Check whether to continue and calculate next pose
        T_tcp_in_world, current_pose = self._getStickoutInWorld()
        current_position = current_pose.pos
        # check trajectory following if task length is driven
        is_criteria_end_reached, dist_to_end_criteria = end_eval.evaluate(current_position)

        #%% Get target poses
        if self._config.CONTROLLER_CHOICE == 1: #PID
            target_pose, valid_pose, dist_to_end_path = self.pose_generator.getTargetPose(current_position, self._path_data_handler.getData())
        elif self._config.CONTROLLER_CHOICE == 2: #2DoF-PID
            target_pose, valid_pose, dist_to_end_path = self.pose_generator.getTargetPoseLookAhead(current_position, self._path_data_handler.getData())
        else:
            rospy.logwarn("No controller is selected. Choose PID or 2DoF-Controller")
            return False, 0
        rospy.loginfo_throttle(1, f"{self.PREFIX} Seam Arc Length: {(end_eval.length * 1000):5.1f} mm,"
                               f" distance to end path: {(dist_to_end_path * 1000):5.1f} mm")

        #%% Hard stop if any of these criteria applies
        is_dist_to_end_path_reached = dist_to_end_path < 0
        is_pose_invalid = not valid_pose
        if (dist_to_end_criteria) < 0:
            is_pose_invalid = True
        end_reached = is_dist_to_end_path_reached or is_criteria_end_reached or is_pose_invalid
        if end_reached:
            end_reached_log = (
                f"{self.PREFIX} Stop at {np.linalg.norm(twist_tcp[0]) * 1000:.2f} mm/s. "
                f"Possible reasons: is_dist_to_end_path_reached={is_dist_to_end_path_reached}; "
                f"is_criteria_end_reached={is_criteria_end_reached}; "
                f"is_pose_invalid={is_pose_invalid}"
            )
            rospy.logwarn(end_reached_log)
            return False, dist_to_end_path

        #%% Calculate speed from pose via controller and check whether controller indicates termination
        if self._config.CONTROLLER_CHOICE == 1:
            target_stickout = self.__transformWorldToStickout(T_tcp_in_world, target_pose) # type: ignore
        if self._config.CONTROLLER_CHOICE == 2:
            target_stickout = []
            for i in range(0, len(target_pose)): # type: ignore
                target_stickout.append(self.__transformWorldToStickout(T_tcp_in_world, target_pose[i])) # type: ignore

        # End distance is either from criteria or from available points
        dist_to_end = min([dist_to_end_criteria, dist_to_end_path])
        # Determine whether to trigger deceleration
        trigger_deceleration = dist_to_end <= deceleration_distance
        # Get velocity from controller
        twist_stickout, continueMovement = realtime_controller.getVelocityControllerOutput(target_stickout, trigger_deceleration) # type: ignore

        #%% Stop trajectory following if deceleration ramp finished
        if not continueMovement:
            return False, dist_to_end
        #%% Transform to control frame to move
        twist_tcp = ros_util.transformMotion(twist_stickout, self._T_stickout_to_tcp)
        self.__sendTwist(*twist_tcp)
        return True, dist_to_end

    def __continueMotion(self, n: int):
        """
        """
        status_ok = self.__checkRobotState() and self.__checkRosState() and self.__checkServoState(n)
        return status_ok

    def __checkRobotState(self) -> bool:
        """Check if ROS Control is running on UR

        Returns:
            bool: True if running
        """
        if not self._robot_program_running:
            rospy.logwarn(
                "Follow task stopped because IPA Adapt is not running.")
        return self._robot_program_running

    def __checkRosState(self) -> bool:
        """Check if ROS is running

        Returns:
            bool: True if running
        """
        valid = not rospy.is_shutdown()
        if not valid:
            rospy.logwarn("Follow task stopped because ROS shutdown.")
        return valid

    def __checkServoState(self, n: int = 0) -> bool:
        """Check if Servo has any errors. First 30 iterations are always true

        Args:
            n (int, optional): Running iteration. Defaults to None.

        Returns:
            bool: True if no errors
        """
        if n and n < 30:  # Grace period during start
            servo_status_ok = True  # Always valid
        else:
            servo_status_ok = self.__servo_status == 0
            if not servo_status_ok:
                rospy.logwarn(
                    "Follow task stopped because of MoveIt servo status error")

        return servo_status_ok

    def __transformWorldToStickout(self, T_tcp_in_world: np.ndarray, target: PoseTuple) -> PoseTuple:
        """Transform from world to stickout

        Args:
            T_tcp_in_world (np.array(4x4)): Homogeneous Transformation from world to tcp
            target (PoseTuple): Position xyz, Orientation as quaternion

        Returns:
            tuple: pos, ori
        """

        # Transform to control frame
        affine_tf = py_util.invertAffineTransform(T_tcp_in_world)
        pose = affine_tf @ target.asArray()
        # Transform to TCP Rotated frame
        pose = self._T_tcp_to_stickout @ pose
        # Split into tuple
        return PoseTuple.fromArray(pose)

    def __sendTwist(self, v=[0, 0, 0], w=[0, 0, 0]) -> None:
        self.__pub_twist.publish(
            ros_util.createTwist(self._config.TF_CONTROL, v, w))


#%%
if __name__ == "__main__":
    pass
