#!/usr/bin/env python3
import os
import sys

from processit_core.line import Line
from processit_core.plane import Plane

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

import numpy as np
from scipy.spatial.transform import Rotation
from geometry_msgs.msg import Transform


def transformPointsByTransform(points: np.ndarray, transform: Transform) -> np.ndarray:
    """
        Transforms a set of points by a given transform
    """
    translation = np.array([transform.translation.x, transform.translation.y, transform.translation.z])
    rotation_in_quat = np.array([transform.rotation.x, transform.rotation.y, transform.rotation.z, transform.rotation.w])
    rotation_matrix = Rotation.from_quat(rotation_in_quat).as_matrix()
    rotated_points = rotation_matrix @ points.T    # Matrix multiplication
    transformed_points = rotated_points.T + translation
    return transformed_points

def computeCentroid(points: np.ndarray) -> np.ndarray:
    """
        Computes the centroid of a set of points
    """
    if points.size == 0:
            raise ValueError("No points to compute centroid")
    centroid = np.mean(points, axis=0)
    return centroid

def angleBetweenTwoVectors(vector1: np.ndarray, vector2: np.ndarray) -> float:
    """
        Calculates the angle between two vectors in degrees
    """
    if np.linalg.norm(vector1) == 0 or np.linalg.norm(vector2) == 0:
        return 0
    cos_angle = np.clip(np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2)), -1.0, 1.0)
    angle = np.arccos(cos_angle)
    angle_in_degrees = np.degrees(angle)
    return angle_in_degrees

def checkTwoVectorsParallel(vector1: np.ndarray, vector2: np.ndarray, degree_tolerance: float = 20) -> bool:
    """
        Checks if two vectors are parallel within a given tolerance
    """
    # check if vectors are zero vectors
    if np.linalg.norm(vector1) == 0 or np.linalg.norm(vector2) == 0:
        return False
    vector1 = vector1 / np.linalg.norm(vector1)
    vector2 = vector2 / np.linalg.norm(vector2)
    cos_angle = np.clip(np.dot(vector1, vector2), -1.0, 1.0)
    degree_angle = np.degrees(np.arccos(cos_angle))
    return degree_angle < degree_tolerance or abs(degree_angle - 180) < degree_tolerance

def orientatePlaneToPoint(plane: Plane, point: np.ndarray) -> Plane:
    """
        Orientates a plane to a given point
    """
    if plane is None:
        raise ValueError("Plane is not defined")

    if not bool(plane) or np.linalg.norm(plane.normal) == 0:
        return plane

    a, b, c = plane.normal
    d = plane.d
    if a * point[0] + b * point[1] + c * point[2] + d > 0:
        return plane
    return Plane(np.array([-a, -b, -c, -d]))

def distanceFromPointToPlane(point: np.ndarray, plane: Plane) -> float:
    """
        Calculates the shortest (perpendicular) distance of a point to a plane
    """
    if not bool(plane) or np.linalg.norm(plane.normal) == 0:
        raise ValueError("Plane is not defined")

    a, b, c = plane.normal
    d = plane.d
    distance = abs(a * point[0] + b * point[1] + c * point[2] + d) / np.sqrt(a**2 + b**2 + c**2)
    return distance

def distanceFromPointToPlaneInDirection(point: np.ndarray, plane: Plane, direction: np.ndarray) -> float:
    """
        Calculates the distance of a point to a plane in a given direction
    """
    if not bool(plane) or np.linalg.norm(plane.normal) == 0:
        raise ValueError("Plane is not defined")

    x0, y0, z0 = point
    a, b, c = plane.normal
    d = plane.d
    dx, dy, dz = direction
    # Calculate the parameter t for the intersection point
    t = -(a * x0 + b * y0 + c * z0 + d) / (a * dx + b * dy + c * dz)
    # Calculate the intersection point
    intersection_point = np.array([x0 + t * dx, y0 + t * dy, z0 + t * dz])
    # Calculate the distance between the original point and the intersection point
    distance = np.linalg.norm(intersection_point - point)
    return float(distance)

def projectPointsToPlane(points: np.ndarray, plane: Plane) -> np.ndarray:
    """
        Projects a set of points to a plane
    """
    if not bool(plane) or np.linalg.norm(plane.normal) == 0:
        raise ValueError("Plane is not defined")

    a, b, c = plane.normal
    d = plane.d
    projected_points = points - (a * points[:, 0] + b * points[:, 1] + c * points[:, 2] + d)[:, np.newaxis] * np.array([a, b, c])
    return projected_points

def distanceFromPointToLine(point: np.ndarray, line: Line) -> float:
    """
    Calculates the shortest (perpendicular) distance of a point to a line
    """
    if np.linalg.norm(line.direction) == 0:
        raise ValueError("Line direction is zero vector")

    distance = np.linalg.norm(
        np.cross(line.direction, line.point - point)
    ) / np.linalg.norm(line.direction)
    return distance

def intersectionPlaneAndLine(plane: Plane, line: Line) -> np.ndarray:
    """
    Calculates the intersection point of a plane and a line
    """
    if np.linalg.norm(line.direction) == 0 or np.linalg.norm(plane.normal) == 0:
        raise ValueError("Line direction or plane normal is zero vector")

    a, b, c = plane.normal
    d = plane.d
    x0, y0, z0 = line.point
    dx, dy, dz = line.direction
    t = -(a * x0 + b * y0 + c * z0 + d) / (a * dx + b * dy + c * dz)
    intersection_point = np.array([x0 + t * dx, y0 + t * dy, z0 + t * dz])
    return intersection_point

def interpolatePoint(from_point: np.ndarray, to_point: np.ndarray, interpolate_param: float) -> np.ndarray:
    """
    Interpolates a point between two points
    """
    interpolate_param = np.clip(interpolate_param, 0, 1) # Clip the interpolation parameter to the range [0, 1]
    interpolated_point = (1 - interpolate_param) * from_point + interpolate_param * to_point
    return interpolated_point

#%%
if __name__ == "__main__":
    pass
