#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 14 12:45:34 2021

@author: jonasrajagopal
"""

import cv2
import numpy as np
import statistics
import time

starttime = time.time()

points =[[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
original_points = points

#Region - the maximum distance of KeyPoints that are tracked. 
#Example: With region = 150, all points within a 300x300 pixel square centered on the point in question are tracked
region = 150

#Percentile - the percentile of the motions of the keypoints used to approximate the motion of the point in question. 50 means that the median is used.
percentile = 50


cap = cv2.VideoCapture("./data/video/input.mp4")
_, frame = cap.read()
old_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

lk_params = dict(winSize = (10,10), 
                 maxLevel = 2, 
                 criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))


fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('./Sift_detections/output.mp4', fourcc, 30.0, (1280,720))
count = 0

while True:
    ret, frame = cap.read()
    start_time = time.time()
   
    if ret:
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        print('Video has ended or failed, try a different video format!')
        break
        
    #Find the new SIFT keypoints every 10 frames 
    if count % 10 == 0:
        sift = cv2.xfeatures2d.SIFT_create()
        kp = sift.detect(gray_frame,None)
        pts = cv2.KeyPoint_convert(kp)
        pts1, pts2, pts3, pts4 = [], [], [], []
        for i in range(0,len(pts)):
            if (abs(pts[i][0] - points[0][0]) < region and abs(pts[i][1] - points[0][1]) < region):
                pts1.append(pts[i])
            if (abs(pts[i][0] - points[1][0]) < region and abs(pts[i][1] - points[1][1]) < region):
                pts2.append(pts[i])
            if (abs(pts[i][0] - points[2][0]) < region and abs(pts[i][1] - points[2][1]) < region):
                pts3.append(pts[i])
            if (abs(pts[i][0] - points[3][0]) < region and abs(pts[i][1] - points[3][1]) < region):
               pts4.append(pts[i])
        
        #old_points = np.array(pts, dtype = np.float32)
        old_points1 = np.array(pts1, dtype = np.float32)
        old_points2 = np.array(pts2, dtype = np.float32)
        old_points3 = np.array(pts3, dtype = np.float32)
        old_points4 = np.array(pts4, dtype = np.float32)
        
    #Draw the red circles representing where the points were are the beginning of the video.
    for i in range(0, 4):
        cv2.circle(frame, (original_points[i][0], original_points[i][1]), 5, (0,0,255), 2)
    
    #Calculate the optical flow at the relevant SIFT keypoints
    new_points1, status, error = cv2.calcOpticalFlowPyrLK(old_gray, gray_frame, old_points1, None, **lk_params)
    new_points2, status, error = cv2.calcOpticalFlowPyrLK(old_gray, gray_frame, old_points2, None, **lk_params)
    new_points3, status, error = cv2.calcOpticalFlowPyrLK(old_gray, gray_frame, old_points3, None, **lk_params)
    new_points4, status, error = cv2.calcOpticalFlowPyrLK(old_gray, gray_frame, old_points4, None, **lk_params)
    
    #Compute the median of all of the optical flows.
    diffs = [[[],[]],[[],[]],[[],[]],[[],[]]]
    for i in range(0,len(new_points1)):
        diffs[0][0].append(new_points1[i][0] - old_points1[i][0])
        diffs[0][1].append(new_points1[i][1] - old_points1[i][1])
        
    for i in range(0,len(new_points2)):
        diffs[1][0].append(new_points2[i][0] - old_points2[i][0])
        diffs[1][1].append(new_points2[i][1] - old_points2[i][1])

        
    for i in range(0,len(new_points3)):
        diffs[2][0].append(new_points3[i][0] - old_points3[i][0])
        diffs[2][1].append(new_points3[i][1] - old_points3[i][1])

    for i in range(0,len(new_points4)):
        diffs[3][0].append(new_points4[i][0] - old_points4[i][0])
        diffs[3][1].append(new_points4[i][1] - old_points4[i][1])
              
    deltas = [[],[],[],[]]
    
    for i in range(0, len(diffs)):
        for j in range(0, len(diffs[i])):
            deltas[i].append(np.percentile(diffs[i][j], percentile))
    for i in range(0, len(points)):
        for j in range(0, len(points[i])):      
            points[i][j] = points[i][j] + deltas[i][j]
            points[i][j] = points[i][j].astype(np.float32)
            
    #Draw the new locations
    for i in range(0, 4):
        cv2.circle(frame, (points[i][0], points[i][1]), 5, (0,255,0), -1)
    
     
   # if count % 10 == 0:
    #    print(points)
    cv2.imshow("Frame", frame)
    out.write(frame)
    
    count = count + 1
    
    old_gray = gray_frame.copy()
    old_points1 = new_points1
    old_points2 = new_points2
    old_points3 = new_points3
    old_points4 = new_points4
   
    #fps = 1.0 / (time.time() - start_time)
    #print("FPS: %.2f" % fps)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

out.release()
cap.release()
cv2.destroyWindow("Frame")
cv2.destroyAllWindows()
cv2.waitKey(1)

print("Time = ", time.time()- starttime)   
