#!/usr/bin/env python3
import os
import sys


sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

class Slit:
    def __init__(self):
        self.__depth = 0
        self.__width = 0
        self.__length = 0
        self.__start_midpoint = None
        self.__end_midpoint = None
        pass

    def setSlitWidth(self, width: float):
        self.__width = width

    def setSlitLength(self, length: float):
        self.__length = length

    def setSlitDepth(self, depth: float):
        self.__depth = depth

    def getSlitWidth(self) -> float:
        return self.__width

    def setStartMidpoint(self, start_midpoint):
        self.__start_midpoint = start_midpoint

    def setEndMidpoint(self, end_midpoint):
        self.__end_midpoint = end_midpoint

    def getSlitLength(self) -> float:
        return self.__length

    def getSlitDepth(self) -> float:
        return self.__depth

    def getStartMidpoint(self):
        return self.__start_midpoint

    def getEndMidpoint(self):
        return self.__end_midpoint

    def isEndDetected(self) -> bool:
        return self.__end_midpoint is not None
