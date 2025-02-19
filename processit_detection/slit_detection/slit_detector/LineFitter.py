#!/usr/bin/env python3
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from processit_core.line import Line


class LineFitter:
    # TODO: how to score the line fitting, R squared?

    @staticmethod
    def fitLineRansac(
        points: np.ndarray, threshold: float, n_iterations=1000, min_n_inliers=10
    ) -> "tuple[Line | None, np.ndarray]":
        """Fit a line to 3D points using RANSAC

        Args:
            threshold (float): Maximum distance for inlier
            n_iterations (int): Number of iterations
            min_n_inliers (int, optional): Minimum number of inliers. Defaults to 10.

        Returns:
            Line: Line object
            List[int]: Indices of inliers
        """
        if len(points) < 2:
            return None, np.zeros(len(points), dtype=bool)

        cls = LineFitter()
        best_inlier_indices = np.zeros(len(points), dtype=bool)
        best_line = None

        for _ in range(n_iterations):
            # Randomly select two points

            sample = points[np.random.choice(len(points), 2, replace=False)]

            # Create a line from these two points
            line = cls.__createLineFromTwoPoints(sample[0], sample[1])
            if line is None:
                continue

            # Calculate distances of all points to this line
            distances = cls.__getDistancePointsLine(points=points, line=line)

            # Find inliers
            inliers = distances < threshold

            # Update best model if we found more inliers
            if np.sum(inliers) > np.sum(best_inlier_indices) and np.sum(inliers) >= min_n_inliers:
                best_inlier_indices = inliers
                best_line = line

        # Refit the line using all inliers
        if best_line is not None:
            best_line = cls.fitLineSvd(points=points[best_inlier_indices])

        return best_line, best_inlier_indices

    @staticmethod
    def fitLineSvd(points: np.ndarray):
        """Fit line to points using SVD.

        Returns:
            Line: Line object
        """
        if len(points) < 2:
            return None

        # Calculate centroid
        centroid = np.mean(points, axis=0)
        # Using SVD to get line direction
        _, _, vh = np.linalg.svd(points - centroid)
        direction = vh[0]
        return Line(centroid, direction)

    ##########################################
    def __createLineFromTwoPoints(self, point_1: np.ndarray, point_2: np.ndarray):
        """Create a line from two points

        Args:
            point_1 (np.ndarray): First point
            point_2 (np.ndarray): Second point

        Returns:
            Line: Line object
        """
        direction = point_2 - point_1
        direction_norm = np.linalg.norm(direction)
        if direction_norm < 1e-6:
            return None
        direction /= direction_norm
        return Line(point_1, direction)

    def __getDistancePointsLine(self, points: np.ndarray, line: Line):
        """Calculate distance between points and line

        Args:
            line (Line): Line object

        Returns:
            np.array(n): Distances between points and line
        """
        vectors = points - line.point
        projection_lengths = np.dot(vectors, line.direction)
        closest_points = line.point + np.outer(projection_lengths, line.direction)
        return np.linalg.norm(points - closest_points, axis=1)
