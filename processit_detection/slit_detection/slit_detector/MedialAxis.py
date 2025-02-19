#!/usr/bin/env python3
import os
import sys


sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import open3d as o3d
import numpy as np

from slit_detection.slit_detector.MedialAxisPoint import MedialAxisPoint

class MedialAxis:
    MIN_SLIT_WIDTH = 1.8e-3 # [m] TODO: pre-defined in config file
    MAX_SLIT_WIDTH = 22e-3 # [m] TODO: pre-defined in config file
    MIN_SEPARATION_ANGLE = 135 # [deg] TODO: pre-defined in config file

    def __init__(self):
        pass

    @staticmethod
    def computeMedialAxis(boundary_pcd: o3d.geometry.PointCloud) -> 'list[MedialAxisPoint]':
        """
        Compute the medial axis of the boundary point cloud using the shrinking circle principles
        """
        cls = MedialAxis()
        medial_axis_points = []
        # Find the maximum distance between boundary points to set the initial radius
        init_radius = cls.MAX_SLIT_WIDTH
        boundary_tree = o3d.geometry.KDTreeFlann(boundary_pcd)
        # Compute the medial axis
        for point_idx in range(len(boundary_pcd.points)):
            # Get the query point and its normal
            query_point = np.asarray(boundary_pcd.points[point_idx])
            query_normal = np.asarray(boundary_pcd.normals[point_idx])

            # Find the medial axis point
            medial_axis_point = cls.__findMedialAxisPoint(query_point, query_normal, init_radius, boundary_pcd, boundary_tree)
            if medial_axis_point is None:
                continue
            # Append the medial axis point to the list
            medial_axis_points.append(medial_axis_point)

        return medial_axis_points

    ##########################################
    def __findMedialAxisPoint(self, query_point: np.ndarray, query_normal: np.ndarray, init_radius: float, boundary_pcd: o3d.geometry.PointCloud, boundary_tree: o3d.geometry.KDTreeFlann):
        """
        Find the medial axis point for the query point
        """
        radius = init_radius
        # Iterate to find the center point ~ medial axis point
        for _ in range(10):
            # Compute the center point
            center_point = self.__computeCenterPoint(query_point, query_normal, radius)
            # Find other boundary points
            [k, idx, _] = boundary_tree.search_radius_vector_3d(center_point, 1.5 * radius)
            other_boundary_points = np.asarray(boundary_pcd.points)[idx]
            # Remove the query point from the other boundary points
            other_boundary_points = other_boundary_points[other_boundary_points != query_point].reshape(-1, 3)
            # Find the closest point on the boundary to the center point but not the query point
            closest_point, distance_center_closest = self.__findClosestPointToCenter(center_point, other_boundary_points)
            if closest_point is None:
                return None
            # Check if the distance between the center point and the closest point is close to the radius
            if np.isclose(distance_center_closest, radius, atol=1e-3):
                # Compute the separation angle
                separation_angle = self.__computeSeparationAngle(center_point, query_point, closest_point)
                # Check if the separation angle has an acceptable value
                if separation_angle > self.MIN_SEPARATION_ANGLE:
                    # Create a new medial axis point
                    medial_axis_point = MedialAxisPoint(
                        center_point=center_point,
                        radius=radius,
                        query_point=query_point,
                        query_normal=query_normal,
                        point_of_contact=closest_point,
                        separation_angle=separation_angle)
                    # Return the medial axis point
                    return medial_axis_point
            # Compute the new radius
            radius = self.__computeNewRadius(query_point, query_normal, closest_point)
            # If the new radius is not within the acceptable range of half the slit width, return
            if radius < (self.MIN_SLIT_WIDTH / 2) or radius > (self.MAX_SLIT_WIDTH / 2):
                return None
        return None

    def __findMaxDistanceBetweenBoundaryPoints(self, boundary_pcd: o3d.geometry.PointCloud) -> float:
        """
        Find the maximum distance between boundary points
        """
        max_distance = 0
        for i in range(len(boundary_pcd.points)):
            for j in range(i + 1, len(boundary_pcd.points)):
                distance = np.linalg.norm(np.asarray(boundary_pcd.points[i]) - np.asarray(boundary_pcd.points[j]))
                if distance > max_distance:
                    max_distance = distance
        return float(max_distance)

    def __computeCenterPoint(self, query_point: np.ndarray, query_normal: np.ndarray, radius: float) -> np.ndarray:
        """
        Compute the center point of the medial axis:
            This center point lies on the line defined by the query point and the query normal
            and is at a distance of radius from the query point
        """
        return np.asarray(query_point + radius * query_normal)

    def __findClosestPointToCenter(self, center_point: np.ndarray, boundary_points: np.ndarray):
        """
        Find the closest point on the boundary to the center point but not the query point
        """
        min_distance = float('inf')
        closest_point = None
        for boundary_point in boundary_points:
            distance = np.linalg.norm(boundary_point - center_point)
            if distance < min_distance:
                min_distance = distance
                closest_point = boundary_point
        return closest_point, min_distance

    def __computeNewRadius(self, query_point: np.ndarray, query_normal: np.ndarray, closest_point: np.ndarray) -> float:
        """
        Compute the new radius to find the new center point
        """
        # Compute the vector from the closest point to the query point
        vector = closest_point - query_point
        # Compute the distance between the query point and the closest point
        distance = np.linalg.norm(vector)
        # Compute the cosine of the angle between the query normal and the vector
        cos_angle = np.dot(query_normal, vector) / distance
        # Compute the new radius
        new_radius = distance / 2 if np.isclose(cos_angle, 0) else distance / (2 * cos_angle)
        return float(new_radius)

    def __computeSeparationAngle(self, center_point: np.ndarray, query_point: np.ndarray, closest_point: np.ndarray) -> float:
        """
        Compute the separation angle between the vectors from the query point and from the closest point to the center point

        Args:
            center_point: The current center point
            query_point: The query point on the boundary
            closest_point: The closest point on the boundary to the center point

        Returns:
            The separation angle in degrees
        """
        vector_query = query_point - center_point
        vector_closest = closest_point - center_point
        norm_query = np.linalg.norm(vector_query)
        norm_closest = np.linalg.norm(vector_closest)
        if np.isclose(norm_query, 0) or np.isclose(norm_closest, 0):
            return 0 # TODO: handle this appropriately

        cos_angle = np.dot(vector_query, vector_closest) / (norm_query * norm_closest)
        cos_angle = np.clip(cos_angle, -1, 1)
        angle = np.degrees(np.arccos(cos_angle))
        return angle
