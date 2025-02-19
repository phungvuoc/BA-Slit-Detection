#!/usr/bin/env python3
import rospy
from slit_detection.DetectionMode import DetectionMode
from slit_detection.ScanLineData import SlitScanLineData
from slit_detection.slit_scan_recorder.SlitScanRecorder import SlitScanRecorder
from slit_detection.SlitFollower import SlitFollower
from std_srvs.srv import Trigger
from std_srvs.srv import TriggerRequest
from std_srvs.srv import TriggerResponse

from processit_adapt.adapt_status import STATUS
from processit_adapt.configs import config
from processit_adapt.configs.config_base import GeometryType
from processit_core import py_util
from processit_core import ros_util
from processit_core.pose_tuple import PoseTuple
from processit_msgs.msg import GeometricFeature
from processit_msgs.msg import ScanData
from processit_msgs.srv import ReadDetectionAbortReason
from processit_msgs.srv import ReadDetectionAbortReasonRequest
from processit_msgs.srv import ReadDetectionAbortReasonResponse
from processit_msgs.srv import SetLaserParameter
from processit_msgs.srv import SetLaserParameterRequest
from processit_msgs.srv import StartDetection
from processit_msgs.srv import StartDetectionRequest
from processit_msgs.srv import StartDetectionResponse

class SlitDetection:
    LOGGING_ACTIVE = False # change this to False if don't want to log the slit scan data

    def __init__(self) -> None:
        """
        Initialize GroundPlaneTeacher instance.
        """
        self.PREFIX = "[SlitDetection]"
        self._init_params()
        self._initSubcribers()
        self._initPublishers()
        self._initServices(py_util.createPrefix("processit_detection", "slit_detection"))

        self.__setLaserMode(1) # Reduce laser power

        rospy.logwarn(f"----- {self.PREFIX} ready ----- ")

    def _init_params(self):
        """
        Initialize parameters.
        """
        # Indicates whether simulation mode is enabled when mode > 0
        self.sim = int(rospy.get_param("/processit/mode")) > 0 # type: ignore
        # Path to the package to handle error text
        self.package_path = ros_util.getPackagePath("processit_adapt")
        # get configuration
        self.config = config.init(GeometryType.SLIT, "processit_adapt")
        # IP address of the robot to set laser mode if not in simulation mode
        self.robot_ip = rospy.get_param("/ur_hardware_interface/robot_ip", "192.168.2.2")
        # Service proxy for setting scanner parameters or laser mode
        self.srv_set_scanner_parameters = rospy.ServiceProxy("/processit_sensors/ScannerHandler/SetLaserParameter", SetLaserParameter)

        self.mode = DetectionMode.ABORTED
        self.follower = SlitFollower(detector_type="slit", config=self.config)
        self.recorder = SlitScanRecorder(SlitDetection.LOGGING_ACTIVE)

    def _initSubcribers(self):
        self.pointcloud_sub = rospy.Subscriber("/scan_world", ScanData, self.incomingSlitScanDataCallback)

    def _initPublishers(self):
        self.task_point_publisher = rospy.Publisher(
            "processit_detection/online_feature_detection/taskpoints",
            GeometricFeature,
            queue_size=1000,
        )

    def _initServices(self, prefix: str):
        rospy.Service(prefix + "startSlitDetection", StartDetection, self.startSlitDetectionCallback)
        rospy.Service(prefix + "endSlitDetection", Trigger, self.endSlitDetectionCallback)
        rospy.Service(prefix + "pauseSlitDetection", Trigger, self.pauseSlitDetectionCallback)
        rospy.Service(prefix + "resumeSlitDetection", Trigger, self.resumeSlitDetectionCallback)
        rospy.Service(prefix + "abortSlitReason", ReadDetectionAbortReason, self.abortSlitReasonCallback)

    ##########################################
    # Services
    ##########################################
    def startSlitDetectionCallback(self, req: StartDetectionRequest):
        """
        Start the slit detection process.
        """
        if req.geometry != "slit":
            return self.__handleReturn(StartDetectionResponse, False, STATUS.GEOMETRY_NOT_SET)

        rospy.logwarn(f"{self.PREFIX} --- Start slit detection ---")

        self.follower.setTaskPointPublisher(self.task_point_publisher)
        self.follower.setViewFieldLimits(req.search_field_upper_limit, req.search_field_lower_limit)

        self.__setLaserMode(2) # Turn laser power to full
        self.recorder.startSlitRecording()
        self.__switchToSearchingMode()

        return self.__handleReturn(StartDetectionResponse, True, STATUS.NONE)

    def endSlitDetectionCallback(self, req: TriggerRequest):
        """
        End the slit detection process.
        """
        success = True
        status = STATUS.NONE

        if self.follower is None:
            rospy.logwarn(f"{self.PREFIX} No slit detection process to end.")
            success = False
            status = STATUS.DEBUG_ERROR
            return self.__handleReturn(TriggerResponse, success, status)

        rospy.logwarn(f"{self.PREFIX} --- End slit detection ---")

        self.__switchToAbortedMode()

        self.__setLaserMode(1) # Reduce laser power
        self.recorder.endSlitRecording()
        return self.__handleReturn(TriggerResponse, success, status)

    def pauseSlitDetectionCallback(self, req: TriggerRequest):
        """
        Pause the slit detection process.
        """
        if self.follower is None:
            rospy.logwarn(f"{self.PREFIX} No slit detection process to pause.")
            return self.__handleReturn(TriggerResponse, False, STATUS.NONE)
        if self.mode == DetectionMode.ABORTED:
            rospy.logwarn(f"{self.PREFIX} Tried to call pause detection, when detection was previously not started or aborted.")
            return self.__handleReturn(TriggerResponse, False, STATUS.NONE)

        rospy.logwarn(f"{self.PREFIX} Pause slit detection.")

        self.__setLaserMode(1) # Reduce laser power
        self.__switchToPausingMode()
        return self.__handleReturn(TriggerResponse, True, STATUS.NONE)

    def resumeSlitDetectionCallback(self, req: TriggerRequest):
        """
        Resume the slit detection process.
        """
        if self.follower is None:
            rospy.logwarn(f"{self.PREFIX} No slit detection process to end.")
            return self.__handleReturn(TriggerResponse, False, STATUS.NONE)
        if self.mode == DetectionMode.ABORTED:
            rospy.logwarn(f"{self.PREFIX} Tried to call resume detection, when detection was previously not started or aborted.")
            return self.__handleReturn(TriggerResponse, False, STATUS.NONE)

        rospy.logwarn(f"{self.PREFIX} Resume slit detection.")

        self.__setLaserMode(2) # Turn laser power to full
        self.__switchToFollowingMode()
        return self.__handleReturn(TriggerResponse, True, STATUS.NONE)

    def abortSlitReasonCallback(self, req: ReadDetectionAbortReasonRequest):
        """
        Abort the slit detection process.
        """
        rospy.logwarn(f"{self.PREFIX} Abort slit detection: Not implemented.")
        success = True
        status = STATUS.NONE
        return self.__handleReturn(ReadDetectionAbortReasonResponse, success, status)

    ##########################################
    # Processing
    ##########################################
    def incomingSlitScanDataCallback(self, msg: ScanData):
        """
        Callback for incoming scan data.
        """
        if (
            self.mode == DetectionMode.ABORTED
            or self.mode == DetectionMode.PAUSING
            or self.follower is None
        ):
            return

        # get scan data
        new_scan_line_data = SlitScanLineData(
            scan_line_points=ros_util.pointCloudToArray(msg.cloud),
            sensor_to_world=msg.tf_to_world.transform,
            world_to_sensor=msg.tf_to_sensor.transform,
            sensor_pose=PoseTuple.fromGeomMsg(msg.tf_to_world),
        )
        # follower process
        success = self.follower.processScanLine(new_scan_line_data)
        if not success:
            self.__switchToAbortedMode()
        # TODO: aborted by detector

    ##########################################
    # Functions
    ##########################################
    def __handleReturn(self, responseType, success: bool, status: STATUS, *args):
        message = status.lookupErrorText('en', self.package_path) #TODO: lang_code???
        if not success:
            rospy.logerr(f"{self.PREFIX} {message}")
        return responseType(success, message, *args)

    def __setLaserMode(self, lasermode: int, exposure: int = 200, frequency: int = 130):
        """
        Set the laser parameters.
        """
        if not self.sim and self.robot_ip != "127.0.0.1":
            self.srv_set_scanner_parameters(SetLaserParameterRequest(
                exposure=exposure,
                frequency=frequency,
                lasermode=lasermode))

    def __switchToSearchingMode(self):
        # Can only switch to searching mode from not started or aborted mode
        if self.mode != DetectionMode.ABORTED:
            return
        rospy.loginfo(f"{self.PREFIX} Switching to searching mode.")
        self.mode = DetectionMode.SEARCHING
        if self.follower is None:
            raise ValueError("Follower not initialized")
        self.follower.updateMode(self.mode)

    def __switchToPausingMode(self):
        # Can only switch to pausing mode from searching or following mode
        if (
            self.mode != DetectionMode.SEARCHING
            and self.mode != DetectionMode.FOLLOWING
        ):
            return
        rospy.loginfo(f"{self.PREFIX} Switching to pausing mode.")
        self.mode = DetectionMode.PAUSING
        if self.follower is None:
            raise ValueError("Follower not initialized")
        self.follower.updateMode(self.mode)

    def __switchToFollowingMode(self):
        # Can only switch to following mode from pausing mode
        if self.mode != DetectionMode.PAUSING:
            return
        rospy.loginfo(f"{self.PREFIX} Switching to following mode.")
        self.mode = DetectionMode.FOLLOWING
        if self.follower is None:
            raise ValueError("Follower not initialized")
        self.follower.updateMode(self.mode)

    def __switchToAbortedMode(self):
        # Do not need to switch to aborted mode if already in aborted mode
        if self.mode == DetectionMode.ABORTED:
            return
        rospy.loginfo(f"{self.PREFIX} Switching to aborted mode.")
        self.mode = DetectionMode.ABORTED
        if self.follower is None:
            raise ValueError("Follower not initialized")
        self.follower.updateMode(self.mode)


# %%
if __name__ == "__main__":
    rospy.init_node("SlitDetection")
    # Get ProcessIt mode and set sim mode
    try:
        SlitDetection()
        rospy.spin()
    except Exception as e:
        rospy.logerr(f"Could not find param: {e}. Terminate SlitDetection!")
        quit()
