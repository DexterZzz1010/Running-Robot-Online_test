import cv2
import cv2 as cv
import numpy as np
import os
import pandas as pd
import csv
import timeit
# import imutils
import json
import base64
from itertools import islice


def detect(readimg):
    img = cv.bilateralFilter(readimg, 21, 75, 75)
    # img = cv.medianBlur(img, 3)
    img = cv2.cvtColor(img, code=cv2.COLOR_BGR2HSV)  # 颜色空间的转变

    lower_green = np.array([20, 15, 46])
    upper_green = np.array([90, 255, 255])

    mask = cv2.inRange(img, lower_green, upper_green)

    res = cv2.bitwise_and(readimg, readimg, mask=mask)

    gray = cv.cvtColor(res, cv.COLOR_BGR2GRAY)
    gray = cv.bilateralFilter(gray, 21, 75, 75)
    gray = cv.medianBlur(gray, 3)

    kernel = np.ones((3, 3), np.uint8)
    erosion = cv2.erode(gray, kernel, iterations=1)

    edges = cv.Canny(erosion, 30, 150)  # 边缘检测

    contours, _ = cv2.findContours(
        edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  # 轮廓检测

    def myApprox(contour):
        num = 0.01
        epsilon = num * cv.arcLength(contour, True)
        contour = cv.approxPolyDP(contour, epsilon, True)

        while (1):
            if len(contour) <= 4:
                break
            else:
                num = num * 1.5
                epsilon = num * cv2.arcLength(contour, True)
                contour = cv2.approxPolyDP(contour, epsilon, True)
                continue
        return contour

    contours = sorted(contours, key=cv2.contourArea, reverse=True)  # 轮廓排序

    contour = myApprox(contours[0])
    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)
    box = np.int0(box)

    box = sorted(box, key=lambda x: x[1])
    line = sorted(box[-2:], key=lambda x: x[0])
    return np.sqrt(np.power(line[0] - line[1], 2).sum()), line[0][0], line[0][1]
