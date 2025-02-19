#!/usr/bin/env python3
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import rospy
import numpy as np
import open3d as o3d
import slit_detector.SlitDetectorMath as math_utils
from processit_core.plane import Plane
from processit_core.line import Line
from processit_core.pose_tuple import PoseTuple
from slit_detector.BaseSlitDetector import BaseSlitDetector
from slit_detector.MedialAxis import MedialAxis
from slit_detector.MedialAxisPoint import MedialAxisPoint
from slit_detection.slit_detector.LineFitter import LineFitter
from slit_detection.ScanLineData import SlitScanLineData
from slit_detection.TargetPoint import TargetPoint
from slit_detection.DetectionOutput import DetectionResult, DetectionOutput

class SearchingSlitDetector(BaseSlitDetector):
    PREFIX = "[SearchingSlitDetector]"

    def __init__(self) -> None:
        super().__init__()

    def detect(self, scan_group: 'list[SlitScanLineData]') -> DetectionOutput:
        start_time = rospy.Time.now()
        detection_output = DetectionOutput(
            result=DetectionResult(),
            success=False,
            continue_flag=False
        )
        # %% check length of scan_group
        if len(scan_group) < 2:
            rospy.logwarn(f"{self.PREFIX} Scan group too short to detect slit")
            # TODO: abort detection
            return detection_output

        # get sensor poses of the first and last scan lines
        first_sensor_pose = scan_group[0].sensor_pose
        last_sensor_pose = scan_group[-1].sensor_pose
        if np.dot(first_sensor_pose.y_dir, last_sensor_pose.y_dir) < 0:
            rospy.logerr(f"{self.PREFIX} First and last sensor poses are not in the same ")
            return detection_output
        if not math_utils.checkTwoVectorsParallel(first_sensor_pose.y_dir, last_sensor_pose.y_dir, degree_tolerance=45):
            rospy.logerr(f"{self.PREFIX} First and last y directions of sensor pose are not parallel")
            return detection_output

        # %% create point cloud from scan group
        pcd = self._createPointCloudFromScanGroup(scan_group)

        # %% pre-process scan group to get main plane point cloud
        main_plane_inliers = self._preProcess(pcd, first_sensor_pose, last_sensor_pose)

        # %% process main plane point cloud to get boundary point cloud and medial axis
        boundary_pcd, medial_axis = self._process(main_plane_inliers)

        # %% calculate target points from boundary point cloud and medial axis
        target_points = self._findTargetPoints(
            boundary_pcd, medial_axis, first_sensor_pose, last_sensor_pose
        )
        rospy.loginfo(f"{self.PREFIX} Found {len(target_points)} target points")
        if not self.__validateTargetPoints(target_points):
            rospy.logerr(f"{self.PREFIX} Not enough target points")
            return detection_output

        # Update the detection result
        detection_result = DetectionResult(
            main_plane=self.main_plane,  # type: ignore
            slit_medial_axis_line=self.slit_medial_axis_line,  # type: ignore
            slit=self.slit,  # type: ignore
            target_points=target_points,  # type: ignore
        )

        detection_output = DetectionOutput(
            result=detection_result,
            success=True,
            continue_flag=not self.slit.isEndDetected(), # TODO
        )
        # For evaluation purposes
        end_time = rospy.Time.now()
        rospy.loginfo(f"{self.PREFIX} Detection time: {(end_time - start_time).to_sec():.2f} s")

        first_z_dir = first_sensor_pose.z_dir
        angle_z_dir_normal = math_utils.angleBetweenTwoVectors(first_z_dir, self.main_plane.normal)
        if angle_z_dir_normal < 90:
            projection_anlge = 90 - angle_z_dir_normal
        else:
            projection_anlge = angle_z_dir_normal - 90
        rospy.logerr(f"{self.PREFIX} Angle between z direction and normal of main plane: {angle_z_dir_normal:.2f} degrees")
        rospy.logerr(f"{self.PREFIX} Angle between z direction and main plane: {projection_anlge:.2f} degrees")
        return detection_output

    ##########################################
    def _createPointCloudFromScanGroup(self, scan_group: "list[SlitScanLineData]"):
        # create point cloud from scan group
        point_cloud = o3d.geometry.PointCloud()
        points = np.vstack([scan_line.scan_line_points for scan_line in scan_group])
        point_cloud.points = o3d.utility.Vector3dVector(points)
        # for evaluation purposes
        rospy.logerr(f"{self.PREFIX} Number of scan lines: {len(scan_group)}")
        rospy.logerr(f"{self.PREFIX} Number of points in point cloud: {len(points)}")
        return point_cloud

    ##########################################
    def _preProcess(self, pcd: o3d.geometry.PointCloud, first_sensor_pose: PoseTuple, last_sensor_pose: PoseTuple):
        # %% downsample point cloud
        pcd = pcd.voxel_down_sample(voxel_size=0.0005) # [m] TODO: find appropriate values

        # %% cluster point cloud
        cluster_labels = np.array(
            pcd.cluster_dbscan(eps=0.01, min_points=10)
        )  # TODO: find appropriate values
        if max(cluster_labels) < 0:
            rospy.logwarn(f"{self.PREFIX} No clusters found")
            return pcd

        # %% find main plane
        main_plane, main_plane_inliers, cluster_outliers = (
            self.__findMainPlaneFromClusters(
                pcd, cluster_labels, first_sensor_pose, last_sensor_pose
            )
        )
        # Orientate the plane to the last sensor pose
        main_plane = math_utils.orientatePlaneToPoint(main_plane, last_sensor_pose.pos)
        self.main_plane = main_plane

        # %% find depth of slit
        # find a plane that is parallel to the main plane and fits to the outliers
        if len(cluster_outliers) != 0:
            depth_plane = self._fitPlaneToPointsWithNormal(cluster_outliers.tolist(), main_plane.normal)
            main_plane_point = np.mean(main_plane_inliers, axis=0)
            slit_depth = math_utils.distanceFromPointToPlane(main_plane_point, depth_plane)
            self.slit.setSlitDepth(slit_depth)
            rospy.loginfo(f"{self.PREFIX} Slit depth: {slit_depth * 1000:.2f} mm")

        # %% maintain point cloud only with inliers projected to the main plane
        projected_inliers = math_utils.projectPointsToPlane(
            main_plane_inliers, main_plane
        )

        return projected_inliers

    def __findMainPlaneFromClusters(
        self,
        pcd: o3d.geometry.PointCloud,
        cluster_labels,
        first_sensor_pose,
        last_sensor_pose,
    ):
        main_plane = Plane()
        cluster_inliers = np.array([])
        cluster_outliers = np.array([])

        first_sensor_position = first_sensor_pose.pos
        last_sensor_position = last_sensor_pose.pos
        first_sensor_z_direction = first_sensor_pose.z_dir
        last_sensor_z_direction = last_sensor_pose.z_dir

        distance = float("inf")

        for cluster_label in range(max(cluster_labels) + 1):
            # Get the points in this cluster
            cluster_points = np.asarray(pcd.points)[cluster_labels == cluster_label]
            # If the cluster is too small, skip it
            if len(cluster_points) < 400:
                continue

            # Create a point cloud for this cluster
            cluster_pcd = o3d.geometry.PointCloud()
            cluster_pcd.points = o3d.utility.Vector3dVector(cluster_points)

            # Use RANSAC to fit a plane to the cluster
            plane_model, inlier_indices = cluster_pcd.segment_plane(
                distance_threshold=0.0005, ransac_n=3, num_iterations=1000
            )
            # check if the main plan parallel to x-y-plane of world
            plane = Plane(plane_model)
            world_z_dir = np.array([0,0,1])
            if not math_utils.checkTwoVectorsParallel(plane.normal, world_z_dir):
                continue

            # Calculate the distance of the plane to the first and last sensor pose
            distance_first = math_utils.distanceFromPointToPlaneInDirection(
                first_sensor_position, Plane(plane_model), first_sensor_z_direction
            )
            distance_last = math_utils.distanceFromPointToPlaneInDirection(
                last_sensor_position, Plane(plane_model), last_sensor_z_direction
            )
            mean_distance = (distance_first + distance_last) / 2

            # If the plane is closer to the sensor poses than the current main plane, update the main plane
            if mean_distance < distance:
                distance = mean_distance
                main_plane = Plane(plane_model)
                inlier_indices = np.array(inlier_indices)
                cluster_inliers = np.asarray(cluster_pcd.points)[inlier_indices]
                outlier_indices = np.setdiff1d(np.arange(len(cluster_pcd.points)), inlier_indices)
                cluster_outliers = np.asarray(cluster_pcd.points)[outlier_indices]

        return main_plane, cluster_inliers, cluster_outliers

    ##########################################
    def _process(self, main_plane_inliers: np.ndarray):
        """
        """
        # create point cloud from main plane inliers
        main_plane_pcd = o3d.geometry.PointCloud()
        main_plane_pcd.points = o3d.utility.Vector3dVector(main_plane_inliers)
        main_plane_pcd = main_plane_pcd.voxel_down_sample(voxel_size=0.001) # [m] TODO: find appropriate values

        # find boundary points
        """
        The search radius is crucial to detect boundary points. If the search radius is too small, the boundary points will be missed.
        If the radius is larger then the slit width, the boundary points of slit width can not be detected.
        And also check this radius with the voxel size of down sampling.
        Another problem is the noise in area of start width side of the slit. This noise can influence the detection of start width points.
        """
        boundary_pcd = self._findBoundaryPoints(main_plane_pcd, angle_gap_threshold=5 * np.pi / 12, neighbor_search_radius=0.0025) # [m] TODO: find appropriate values
        
        o3d.visualization.draw_geometries([boundary_pcd]) # type: ignore

        # compute outwards normals of the boundary points
        boundary_pcd = self._computeOutwardsNormals(boundary_pcd, main_plane_pcd)

        # compute the medial axis using the shrinking circle principles
        medial_axis = MedialAxis.computeMedialAxis(boundary_pcd)

        
        center_points = [medial_axis_point.center_point for medial_axis_point in medial_axis]
        center_points_pcd = o3d.geometry.PointCloud()
        center_points_pcd.points = o3d.utility.Vector3dVector(center_points)
        o3d.visualization.draw_geometries([boundary_pcd, center_points_pcd]) # type: ignore
        
        return boundary_pcd, medial_axis

    def _findBoundaryPoints(self, main_plane_pcd: o3d.geometry.PointCloud, angle_gap_threshold: float, neighbor_search_radius: float):
        """ """

        def computeLargestAngleGap(query_point: np.ndarray, neighbors: np.ndarray):
            if neighbors.size < 3:
                raise ValueError(
                    "Not enough neighbors to compute the largest angle gap"
                )

            # vectors from the query point to the neighbors
            vectors = neighbors - query_point
            norms = np.linalg.norm(vectors, axis=1)  # compute norms of vectors
            norms[norms == 0] = 1  # avoid division by zero
            vectors = vectors / norms[:, None]  # normalize vectors
            vectors = vectors.real.astype(
                float
            )  # ensure vectors are real and of type float

            # calculate the angles of each vector relative to a reference direction
            angles = np.arctan2(
                vectors[:, 1], vectors[:, 0]
            )  # Using arctan2 to get the signed angle between vectors and the x-axis
            sorted_angles = np.sort(angles)  # sort angles in ascending order
            angle_diffs = np.diff(
                sorted_angles
            )  # compute the difference between consecutive angles
            wrap_around_diff = 2 * np.pi - (
                sorted_angles[-1] - sorted_angles[0]
            )  # compute the difference between the first and last angle
            angle_diffs = np.append(
                angle_diffs, wrap_around_diff
            )  # append the wrap around difference to the angle differences

            # find the largest angle gap
            max_angle_gap = np.max(angle_diffs)
            return max_angle_gap

        # %% Main function

        boundary_points = []
        main_plane_tree = o3d.geometry.KDTreeFlann(main_plane_pcd)  # create KDTree for point cloud

        for point_idx in range(len(main_plane_pcd.points)):
            # get query point
            query_point = np.asarray(main_plane_pcd.points[point_idx])

            # search for neighbors within a radius
            [k, idx, _] = main_plane_tree.search_radius_vector_3d(query_point, neighbor_search_radius)
            neighbors = np.asarray(main_plane_pcd.points)[idx[1:]]  # Exclude the query point itself

            # compute the largest angle gap
            largest_angle_gap = computeLargestAngleGap(query_point, neighbors)

            # check if the angle gap is larger than the threshold
            if largest_angle_gap > angle_gap_threshold:
                boundary_points.append(query_point)

        boundary_pcd = o3d.geometry.PointCloud()
        boundary_pcd.points = o3d.utility.Vector3dVector(boundary_points)
        return boundary_pcd

    def _computeOutwardsNormals(self, boundary_pcd: o3d.geometry.PointCloud, main_plane_pcd: o3d.geometry.PointCloud):
        """
        Compute the outwards normals of the boundary points that are perpendicular to the boundary neighbors and lie on the main plane
        """

        def computeNormalOfBoundaryPoint(neighbors: np.ndarray):
            """
            compute the normal of the boundary point that is perpendicular to the boundary neighbors and lies on the main plane
            """
            # compute the centroid of the neighbors
            centroid = math_utils.computeCentroid(neighbors)

            # center the neighbors around the centroid
            centered_neighbors = neighbors - centroid

            # compute the covariance matrix
            covariance_matrix = np.cov(centered_neighbors.T)

            # compute the eigenvalues and eigenvectors of the covariance matrix
            eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)

            # find the eigenvector corresponding to the second smallest eigenvalue
            normal = eigenvectors[:, np.argsort(eigenvalues)[1]]

            # normalize the normal vector
            if np.linalg.norm(normal) == 0:
                return normal
            return normal / np.linalg.norm(normal)

        # %% Main function
        boundary_tree = o3d.geometry.KDTreeFlann(boundary_pcd)
        boundary_pcd.normals = o3d.utility.Vector3dVector(np.zeros((len(boundary_pcd.points), 3)))
        main_plane_tree = o3d.geometry.KDTreeFlann(main_plane_pcd)
        main_plane_pcd.normals = o3d.utility.Vector3dVector(np.zeros((len(main_plane_pcd.points), 3)))

        for point_idx in range(len(boundary_pcd.points)):
            # get query point
            query_point = np.asarray(boundary_pcd.points[point_idx])

            # search for a number of boundary neighbors
            [_, idx, _] = boundary_tree.search_knn_vector_3d(query_point, 10)
            boundary_neighbors = np.asarray(boundary_pcd.points)[idx[1:]] # exclude the query point itself

            # compute the normal of the boundary point that is perpendicular to the boundary neighbors and lies on the main plane
            normal = computeNormalOfBoundaryPoint(boundary_neighbors)

            # search for the nearest neighbors in the main plane
            [_, idx, _] = main_plane_tree.search_knn_vector_3d(query_point, 30)
            main_plane_neighbors = np.asarray(main_plane_pcd.points)[idx[1:]] # exclude the query point itself

            # compute the centroid of the main plane neighbors
            centroid = math_utils.computeCentroid(main_plane_neighbors)

            # make sure the normal points outwards the centroid
            if np.dot(normal, centroid - query_point) > 0:
                normal = -normal

            # set the normal of the boundary point
            boundary_pcd.normals[point_idx] = normal

        return boundary_pcd

    ##########################################
    def _findTargetPoints(self, boundary_pcd: o3d.geometry.PointCloud, medial_axis: 'list[MedialAxisPoint]', first_sensor_pose: PoseTuple, last_sensor_pose: PoseTuple) -> 'list[TargetPoint]':
        """
        """
        # %% compute the average y direction from the first and last sensor pose (green line)
        y_direction = (first_sensor_pose.y_dir + last_sensor_pose.y_dir) / 2

        # %% Find the slit medial axis line and the width of the slit
        slit_medial_axis_line, radius = self.__findSlitMedialAxis(medial_axis, y_direction)
        if slit_medial_axis_line is None:
            rospy.logwarn(f"{self.PREFIX} No slit medial axis points found")
            return []
        self.slit_medial_axis_line = slit_medial_axis_line
        slit_width = 2 * radius
        self.slit.setSlitWidth(slit_width)  # [m]
        rospy.loginfo(f"{self.PREFIX} Slit width: {slit_width * 1000:.2f} mm")

        # %% Find the start width midpoint
        start_width_midpoint = self._findWidthMidpoint(type="start", boundary_pcd=boundary_pcd, slit_medial_axis_line=slit_medial_axis_line, radius=radius)
        if start_width_midpoint is None:
            rospy.logwarn(f"{self.PREFIX} No start width points found")
            return []
        self.slit.setStartMidpoint(start_width_midpoint)

        # %% Check if the end width points can be found
        end_width_midpoint = self._findWidthMidpoint(type="end", boundary_pcd=boundary_pcd, slit_medial_axis_line=slit_medial_axis_line, radius=radius)
        if end_width_midpoint is not None:
            rospy.logwarn(f"{self.PREFIX} End width points found")
            self.slit.setEndMidpoint(end_width_midpoint)
            # start_width_midpoint = self.slit.getStartMidpoint()
            slit_length = np.linalg.norm(end_width_midpoint - start_width_midpoint)
            self.slit.setSlitLength(float(slit_length))
            rospy.loginfo(f"{self.PREFIX} Slit length: {slit_length * 1000:.2f} mm")

        # %% calculate target points
        initial_position = (start_width_midpoint + slit_medial_axis_line.direction * self.DEFAULT_START_OFFSET)
        if self.slit.isEndDetected():
            last_position = self.slit.getEndMidpoint() - slit_medial_axis_line.direction * self.DEFAULT_END_OFFSET # type: ignore
        else:
            last_sensor_y_plane = self._getYPlaneOfSensorPose(last_sensor_pose)
            last_position = math_utils.intersectionPlaneAndLine(
                last_sensor_y_plane, slit_medial_axis_line
            )
        orientation = (-1) * self.main_plane.normal
        target_points = self._calculateTargetPoints(start_position=initial_position, last_position=last_position, orientation=orientation)

        return target_points

    def __findSlitMedialAxis(self, medial_axis: 'list[MedialAxisPoint]', y_direction: np.ndarray):
        """
        """
        def getCollinearPointIndices(center_points: 'list[np.ndarray]', direction: np.ndarray):
            """
            """
            collinear_point_idx = []
            for i in range(len(center_points)):
                for j in range(i + 1, len(center_points)):
                    # Compute the vector between the two center points
                    vector = np.asarray(center_points[j]) - np.asarray(center_points[i])
                    # Skip if the vector is too small (points are too close to each other)
                    if np.linalg.norm(vector) < 0.001:
                        continue
                    # Check if the vector is parallel to the direction vector
                    if math_utils.checkTwoVectorsParallel(vector, direction, degree_tolerance=50):
                        if len(collinear_point_idx) == 0:
                            collinear_point_idx.append(i)
                            collinear_point_idx.append(j)
                        elif j not in collinear_point_idx:
                            collinear_point_idx.append(j)
            # Remove duplicates (optional, in case of overlap)
            collinear_point_idx = np.unique(collinear_point_idx, axis=0)
            return collinear_point_idx

        def findSlitMedialAxisPoints(collinear_point_indices: 'list[int]', center_points: 'list[np.ndarray]'):
            slit_medial_axis_points = []
            mean_radius = 0.0

            if len(collinear_point_indices) == 0:
                rospy.logwarn(f"{self.PREFIX} No collinear medial axis points found")
                return slit_medial_axis_points, mean_radius

            # Get the collinear points from the medial axis
            slit_medial_axis_points = [center_points[idx] for idx in collinear_point_indices]

            # Get the radius of the collinear points
            # TODO: check the new mean radius with the previous mean radius in following
            # TODO handle the case: radius of the collinear points are not the same
            collinear_radii = [medial_axis[idx].radius for idx in collinear_point_indices]
            mean_radius = np.mean(collinear_radii)
            return slit_medial_axis_points, mean_radius

        # %% Main function
        normalized_y_direction = y_direction / np.linalg.norm(y_direction)
        # list of center points from the medial axis
        center_points = [medial_axis_point.center_point for medial_axis_point in medial_axis]
        # Find collinear points with the y direction
        # These points are probably the medial axis points of the slit
        collinear_point_indices = getCollinearPointIndices(center_points, normalized_y_direction)
        if len(collinear_point_indices) == 0:
            return None, 0.0

        # Find the slit medial axis points and the radius of the slit
        slit_medial_axis_points, radius = findSlitMedialAxisPoints(collinear_point_indices.tolist(), center_points)
        # Fit a line to the slit medial axis points
        slit_medial_axis_line, _ = LineFitter.fitLineRansac(
            np.array(slit_medial_axis_points), threshold=0.0001, min_n_inliers=2
        )
        if slit_medial_axis_line is None:
            return None, 0.0
        # check if the direction of the slit medial axis line is in the same direction with y direction
        dot_product = np.dot(slit_medial_axis_line.direction, normalized_y_direction)
        if abs(dot_product) < 0.25:
            return None, 0.0
        if dot_product < 0:
            slit_medial_axis_line.direction = -slit_medial_axis_line.direction
        return slit_medial_axis_line, radius

    def _findWidthMidpoint(
        self,
        type: str,
        boundary_pcd: o3d.geometry.PointCloud,
        slit_medial_axis_line: Line,
        radius: float):
        """
        """
        def findWidthPoints(boundary_pcd: o3d.geometry.PointCloud, direction: np.ndarray, line_point: np.ndarray):
            """
            """
            width_points = []
            for point_idx in range(len(boundary_pcd.points)):
                # get query point
                query_point = np.asarray(boundary_pcd.points[point_idx])
                query_normal = np.asarray(boundary_pcd.normals[point_idx])
                # check the normal and the direction
                if np.dot(direction, query_normal) < 0:
                    continue
                # check if distance from the query point to the slit medial axis line is less than the radius
                distance_to_line = math_utils.distanceFromPointToLine(query_point, slit_medial_axis_line)
                # check if the distance to the line is less than the radius minus a small value
                if distance_to_line < radius - 0.0008: # [m] TODO: find appropriate values
                    direction_to_line_point = line_point - query_point
                    if np.linalg.norm(direction_to_line_point) < 0.001:
                        continue
                    direction_to_line_point = direction_to_line_point / np.linalg.norm(direction_to_line_point)
                    # check if the direction to first point is in the same direction with normal
                    if np.dot(direction_to_line_point, query_normal) > 0:
                        width_points.append(query_point)

            return width_points

        # %% Main function
        if type == "start":
            width_plane_normal = slit_medial_axis_line.direction
        elif type == "end":
            width_plane_normal = -slit_medial_axis_line.direction
        else:
            return None
        line_point = slit_medial_axis_line.point
        # %% Find the width points
        width_points = findWidthPoints(boundary_pcd, width_plane_normal, line_point)

        if len(width_points) == 0:
            return None
        # %% Find the width midpoint
        # Find the width plane with normal as the slit medial axis line direction
        width_plane = self._fitPlaneToPointsWithNormal(width_points, width_plane_normal)
        # Find the midpoint of the start width points as the intersection of the start width plane and the slit medial axis line
        width_midpoint = math_utils.intersectionPlaneAndLine(plane=width_plane, line=slit_medial_axis_line)

        return width_midpoint

    ##########################################
    def __validateTargetPoints(self, target_points: 'list[TargetPoint]') -> bool:
        """
        """
        if len(target_points) <= 5:
            return False
        return True

# %%
if __name__ == "__main__":
    pass
