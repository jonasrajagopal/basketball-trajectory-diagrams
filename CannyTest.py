#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  7 13:05:54 2021

@author: jonasrajagopal
"""

import cv2
import numpy as np

img = cv2.imread("Smart1-099.jpg")

imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
imgBlur = cv2.GaussianBlur(imgGray, (3,3), 3)   

imgCanny = cv2.Canny(imgBlur, 200, 300, apertureSize=3)

cv2.imshow("Canny", imgCanny)
cv2.waitKey(1000)  
cv2.destroyAllWindows()