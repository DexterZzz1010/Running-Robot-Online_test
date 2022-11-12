import cv2
import cv2 as cv
import numpy as np
import os
import pandas as pd
import csv
import timeit
import json
import base64
from itertools import islice


def detect(readimg):
    img = cv.bilateralFilter(readimg, 21, 75, 75)
    # img = cv.medianBlur(img, 3)
    img = cv2.cvtColor(img, code=cv2.COLOR_BGR2HSV)  # 颜色空间的转变

    lower_green = np.array([70, 43, 46])
    upper_green = np.array([90, 255, 255])


    mask = cv2.inRange(img, lower_green, upper_green)
    res = cv2.bitwise_and(readimg, readimg, mask=mask)

    gray = cv.cvtColor(res, cv.COLOR_BGR2GRAY)
    gray = cv.bilateralFilter(gray, 21, 75, 75)
    gray = cv.medianBlur(gray, 3)

    kernel = np.ones((3, 3), np.uint8)
    erosion = cv2.erode(gray, kernel, iterations=1)


    edges = cv.Canny(erosion, 30, 150)


    # cv2.imshow('contours.png', edges)
    # cv.waitKey()
    # cv.destroyAllWindows()


    # # 轮廓检测
    # img_contours,contours, _ = cv2.findContours(
    #     edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    index_x = np.where(edges.sum(0) > 0)[0]
    index_y = np.where(edges.sum(1) > 0)[0]

    offset = 3
    if len(index_x) and len(index_y):
        print(index_x[0], index_y[-1])
        x0 = index_x[0] + offset
        x1 = index_x[-1] - offset
        y1 = index_y[-1] - offset
        r,x,y=x1 - x0, x0, y1

        # draw_marker = cv2.drawMarker(readimg, (x, y), (255, 0, 0), cv2.MARKER_CROSS, thickness=3)
        # cv2.imshow('contours.png', draw_marker)
        # cv.waitKey()
        # cv.destroyAllWindows()
        return r,x,y
    else:
        return 0, 0, 0


