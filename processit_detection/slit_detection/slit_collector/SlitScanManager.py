#!/usr/bin/env python3
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

from slit_detection.ScanLineData import SlitScanLineData


class SlitScanManager:
    def __init__(self) -> None:
        self.scan_lines = []

    def addScan(self, scan_line_data: SlitScanLineData):
        self.scan_lines.append(scan_line_data)

    def checkEmpty(self):
        return len(self.scan_lines) == 0

    def getAllScanLines(self):
        if self.checkEmpty():
            return None
        return self.scan_lines

    def getLastScanLine(self):
        if self.checkEmpty():
            return None
        return self.scan_lines[-1]

    def clear(self):
        self.scan_lines.clear()

    def __len__(self):
        return len(self.scan_lines)

    def __getitem__(self, index):
        return self.scan_lines[index]

    def __iter__(self):
        return iter(self.scan_lines)
