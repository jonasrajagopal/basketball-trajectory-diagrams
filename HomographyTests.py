#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 28 20:57:00 2021

@author: jonasrajagopal
"""

import cv2
import numpy as np

#corners = np.array([[488, 313], [807, 343], [729, 445], [366, 406]], np.float32)
#
#basketball_court = np.array([[0,167], [187, 167],[187,323], [0,323]],np.float32)
#points = [[501, 443], [525, 444], [605, 393], [692,433],[720, 448],[770, 585],
#          [803, 558], [916, 598], [828, 438], [657, 318]]
#matrix = cv2.getPerspectiveTransform(corners,basketball_court)
#
#points = np.asarray(points).reshape(-1, 1, 2).astype(np.float64)
#transformed_pts = cv2.perspectiveTransform(points, matrix)    
#transformed_pts = transformed_pts.astype(int).reshape(-1, 2)
#print(transformed_pts)
#court = cv2.imread("./courts/Jazz_court.jpg")
#for i in range(0, len(transformed_pts)):
#    cv2.circle(court, ((transformed_pts[i][0]),(transformed_pts[i][1])), radius=2, color=(0,0,0), thickness = 2)
#
#cv2.imwrite('homographyTest2.jpg',court)
#corners = np.array([[427, 360], [926, 310], [1091, 460], [532, 519]], np.float32)
#basketball_court = np.array([[105,217], [331, 217],[331,416], [105,416]],np.float32)

corners = np.array([[301,260], [787,273], [175,329], [344,429] ], np.float32)
basketball_court = np.array([[0,0], [297, 0],[0,187], [150,356]],np.float32)
points = [[203, 541], [306, 372],[737, 380], [550, 340]]
matrix = cv2.getPerspectiveTransform(corners,basketball_court)

points = np.asarray(points).reshape(-1, 1, 2).astype(np.float64)
transformed_pts = cv2.perspectiveTransform(points, matrix)    
transformed_pts = transformed_pts.astype(int).reshape(-1, 2)
court = cv2.imread("./courts/cavs2.jpg")
for i in range(0, len(transformed_pts)):
    cv2.circle(court, ((transformed_pts[i][0]),(transformed_pts[i][1])), radius=2, color=(0,0,0), thickness = 2)

cv2.imwrite('./homography/homographyTest4.jpg',court)