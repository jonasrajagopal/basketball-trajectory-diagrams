y#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  7 16:19:35 2021

@author: jonasrajagopal
"""

import cv2
import numpy as np
import statistics


#cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture("Sexton2ndAngle.mp4")

_, frame = cap.read()
old_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

lk_params = dict(winSize = (10,10), 
                 maxLevel = 2, 
                 criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))


##def select_point(event, x, y, flags, params):
#  #  global point, point_selected, old_points
#   # if event == cv2.EVENT_LBUTTONDOWN:
#   #     point = (x,y)
#        #old_points = np.array([[x,y]], dtype = np.float32)
#
#        point_selected = True

x, y = 261, 244   
x2, y2 = 916,208
x3,y3 = 430, 359
point_selected = True
point = (x,y)
point2 = (x2,y2)
point3 = (x3, y3)
old_points = np.array([[x,y], [x2,y2], [x3, y3]], dtype = np.float32)
cv2.namedWindow("Frame")
#cv2.setMouseCallback("Frame", select_point)

#point_selected = False
#point = ()
#old_points = np.array([[]])
while True:
    _, frame = cap.read()
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    if point_selected is True:
        cv2.circle(frame, point, 5, (0,0,255), 2)
        cv2.circle(frame, point2, 5, (0,0,255), 2)
    
        new_points, status, error = cv2.calcOpticalFlowPyrLK(old_gray, gray_frame, old_points, None, **lk_params)
        #print(new_points)
        
        diff_x = []
        diff_y = []
        for i in range(0,len(new_points)):
            diff_x.append(new_points[i][0] - old_points[i][0])
            diff_y.append(new_points[i][1] - old_points[i][1])
                
        delta_x = statistics.median(diff_x)
        delta_y = statistics.median(diff_y)
        

        old_gray = gray_frame.copy()
        old_points = new_points
        
        x,y,x2,y2, x3, y3 = new_points.ravel()
        print(type(y2))
        cv2.circle(frame, (x,y), 5, (0,255,0), -1)
        cv2.circle(frame, (x2, y2), 5, (0,255,0), -1)
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()