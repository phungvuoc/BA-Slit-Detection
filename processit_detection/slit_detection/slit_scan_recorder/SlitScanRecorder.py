#!/usr/bin/env python3
import time
from pathlib import Path

import numpy as np
import rospy

from processit_core import py_util
from processit_core import ros_util
from processit_msgs.msg import ScanData

class SlitScanRecorder:
    def __init__(self, logging_active: bool) -> None:
        """
        Initialize SlitScanRecorder instance.
        """
        self.PREFIX = "[SlitScanRecorder]"
        self.LOGGING_ACTIVE = logging_active
        self._init_params()
        self._initServices(py_util.createPrefix("processit_detection", "slit_detection/slit_scan_recorder"))
        self._initSubcribers()
        self._initPublishers()

    def _init_params(self):
        self.slit_scan_data_log_dir: Path = Path(rospy.get_param("SCANDATA_DIR")) # type: ignore
        self.slit_scan_data_log_file = None
        self.slit_scan_data_recording = False
        self.slit_scan_data_recording_sub = None

    def _initServices(self, prefix: str):
        pass

    def _initSubcribers(self):
        pass

    def _initPublishers(self):
        pass

    ##########################################
    # Services
    ##########################################
    def startSlitRecording(self):
        """
        Start recording the slit data.
        """

        if not self.LOGGING_ACTIVE or self.slit_scan_data_log_dir is None:
            return

        file_name = "slit_scan_data_" + time.strftime("%Y%m%d-%H%M%S") + ".npy"
        # file_name = "slit_scan_data.npy"
        self.slit_scan_data_log_file = self.slit_scan_data_log_dir / file_name
        try:
            self.slit_scan_data_log_file.parent.mkdir(exist_ok=True, parents=True)
            self.slit_scan_data_recording = True
            self.slit_scan_data_recording_sub = rospy.Subscriber("/scan_world", ScanData, self.incomingSlitRecordingCallback)
            rospy.logwarn(f"{self.PREFIX} start slit scan data recording.")

        except Exception as e:
            rospy.logerr(f"{self.PREFIX} Failed to start slit scan data recording: {e}")

    def endSlitRecording(self):
        """
        End recording the slit data.
        """
        if not self.LOGGING_ACTIVE or self.slit_scan_data_log_file is None:
            return

        if self.slit_scan_data_recording_sub is not None:
            self.slit_scan_data_recording_sub.unregister()

        self.slit_scan_data_recording = False
        self.slit_scan_data_log_file = None

        rospy.logwarn(f"{self.PREFIX} end slit scan data recording.")

    ##########################################
    # Processing
    ##########################################
    def incomingSlitRecordingCallback(self, msg: ScanData):
        """
        Callback for saving incoming scan data.
        """
        if not self.LOGGING_ACTIVE or self.slit_scan_data_log_file is None or not self.slit_scan_data_recording:
            return

        data = ros_util.pointCloudToArray(msg.cloud)
        #rospy.loginfo(f"{self.PREFIX} tf_to_sensor: {PoseTuple.fromGeomMsg(msg.tf_to_sensor)}")
        #rospy.loginfo(f"{self.PREFIX} tf_to_world: {PoseTuple.fromGeomMsg(msg.tf_to_world)}")
        try:
            if self.slit_scan_data_log_file.exists():
                existing_data = np.load(self.slit_scan_data_log_file, allow_pickle=True)
                # Append the new data to the existing data
                new_data = np.vstack((existing_data, data))
            else:
                # If the file does not exist, the new data is the first entry
                new_data = data

            # Save the new data back to the file
            np.save(self.slit_scan_data_log_file, new_data)

        except Exception as e:
            if self.slit_scan_data_log_file is not None:
                rospy.logerr(f"{self.PREFIX} Failed to save scan data to {self.slit_scan_data_log_file}: \n{e}")

            #self.endSlitRecordingCallback()

#%%
if __name__ == "__main__":
    rospy.logerr("SlitScanRecorder: will be run in SlitDetection, not as standalone node.")
    quit()
