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
#Smart points = [[103, 242], [788,185], [1280, 659], [334, 720]]

#Bojan points = [[580, 242], [1023, 278], [900, 616], [238,508]]

#Lee points = [[871, 380], [1473, 424], [1238, 878], [17, 760]]

#Smart 2 

#points = [[263, 244], [918, 208], [1207, 485], [262, 564]]
#original_points = [[263, 244], [918, 208], [1207, 485], [262, 564]]

#Giannis points = [[426, 314], [981, 341], [151, 512], [966, 572]]

#Kemba points = [[501, 244], [981, 285], [766, 662], [87, 581]]

#PR2 (does not work as well) points = [[299, 272], [812, 239], [1277, 547], [537, 620]]

#PR1 points = [[405, 240], [1028, 271], [918, 719], [23, 654]]

#Horford points = [[193, 201], [731, 156], [1147, 532], [391, 627]]

#Sexton 1 free throw: [[174, 329], [547, 341], [455,432], [13, 416]]

#Smart free throw [[427,358],[929, 310], [1091, 458], [531, 519]]

#Smart 3pt[[464, 234], [809,204], [1223, 576], [813,677]]

#Horford Shot
#points = [[448, 294], [838, 255], [967, 370], [533, 420]]
#original_points = [[448, 294], [838, 255], [967, 370], [533, 420]]

#points = [[429, 360], [927, 311], [102, 241], [531, 520]]
#original_points = [[429, 360], [927, 311], [102, 241], [531, 520]]
points =[[299, 272], [812, 239], [1111, 436], [649, 450]]
original_points = [[299, 272], [812, 239], [1111, 436], [649, 450]]
region = 150
percentile = 50
#points = [[301,260], [175,329], [344,429], [787,273]]
#original_points = [[301,260], [175,329], [344,429], [787,273]]

#Celtics Test 1
#points = [[209,238], [503, 214], [860, 287], [987, 401]]
#original_points = [[209,238], [503, 214], [860, 287], [987, 401]]

##points = [[481, 217], [919, 181], [765, 244], [1043, 286]]
#original_points = [[481, 217], [919, 181], [765, 244], [1043, 286]]
#Celtics 2 Test: 4 corners
#points = [[730, 453], [1043, 286], [1186, 411], [635, 324]]
#original_points = [[730, 453], [1043, 286], [1186, 411], [635, 324]]

#points = [[468, 328], [859, 287], [987, 400], [554, 451]]
#original_points = [[468, 328], [859, 287], [987, 400], [554, 451]]

#points = [[481, 217], [919, 181], [1043, 286], [765, 244]]
#original_points =[[481, 217], [919, 181], [1043, 286], [765, 244]]

#Celtics Test 2  
#points = [[636, 324], [1043, 286], [1188, 411], [730, 454]]
#original_points = [[636, 324], [1043, 286], [1188, 411], [730, 454]]

cap = cv2.VideoCapture("./data/video/PR2.mp4")
_, frame = cap.read()
old_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

lk_params = dict(winSize = (10,10), 
                 maxLevel = 2, 
                 criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))


fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('./Sift_detections/PR2Testing2.mp4', fourcc, 30.0, (1280,720))
count = 0

while True:
    ret, frame = cap.read()
    start_time = time.time()
   
    if ret:
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        print('Video has ended or failed, try a different video format!')
        break
    
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
        

    for i in range(0, 4):
        cv2.circle(frame, (original_points[i][0], original_points[i][1]), 5, (0,0,255), 2)
    
    #new_points, status, error = cv2.calcOpticalFlowPyrLK(old_gray, gray_frame, old_points, None, **lk_params)
    new_points1, status, error = cv2.calcOpticalFlowPyrLK(old_gray, gray_frame, old_points1, None, **lk_params)
    new_points2, status, error = cv2.calcOpticalFlowPyrLK(old_gray, gray_frame, old_points2, None, **lk_params)
    new_points3, status, error = cv2.calcOpticalFlowPyrLK(old_gray, gray_frame, old_points3, None, **lk_params)
    new_points4, status, error = cv2.calcOpticalFlowPyrLK(old_gray, gray_frame, old_points4, None, **lk_params)
    
    
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
              
#    diffs = [[],[]]
#    
#    for i in range(0,len(new_points)):
#        diffs[0].append(new_points[i][0] - old_points[i][0])
#        diffs[1].append(new_points[i][1] - old_points[i][1])
#    
#    deltax = np.percentile(diffs[0], 50)
#    deltay = np.percentile(diffs[1], 50)
    
    deltas = [[],[],[],[]]
    
    for i in range(0, len(diffs)):
        for j in range(0, len(diffs[i])):
            deltas[i].append(np.percentile(diffs[i][j], percentile))
    for i in range(0, len(points)):
        for j in range(0, len(points[i])):      
            points[i][j] = points[i][j] + deltas[i][j]
            points[i][j] = points[i][j].astype(np.float32)
#    for i in range(0, len(points)):
#        points[i][0] += deltax
#        points[i][0] = points[i][0].astype(np.float32)
#    for i in range(0, len(points)):
#        points[i][1] += deltay
#        points[i][1] = points[i][1].astype(np.float32)
    
    for i in range(0, 4):
        cv2.circle(frame, (points[i][0], points[i][1]), 5, (0,255,0), -1)
    
   
        
    
   # if count % 10 == 0:
    #    print(points)
    cv2.imshow("Frame", frame)
    out.write(frame)
    
    count = count + 1
    
    old_gray = gray_frame.copy()
    #old_points = new_points
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

#No lists

#point1 = (x,y)
#point2 = (x2,y2)
#point3 = (x3, y3)
#point4 = (x4, y4)
#
#while True:
#    ret, frame = cap.read()
#    
#    if ret:
#        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#    else:
#        print('Video has ended or failed, try a different video format!')
#        break
#    
#    if count % 10 == 0:
#        sift = cv2.xfeatures2d.SIFT_create()
#        kp = sift.detect(gray_frame,None)
#        pts = cv2.KeyPoint_convert(kp)
#        pts1, pts2, pts3, pts4 = [], [], [], []
#
#        for i in range(0,len(pts)):
#            if (abs(pts[i][0] - x) < 100 and abs(pts[i][1] - y) < 100):
#                pts1.append(pts[i])
#            if (abs(pts[i][0] - x2) < 100 and abs(pts[i][1] - y2) < 100):
#                pts2.append(pts[i])
#            if (abs(pts[i][0] - x3) < 100 and abs(pts[i][1] - y3) < 100):
#                pts3.append(pts[i])
#            if (abs(pts[i][0] - x4) < 100 and abs(pts[i][1] - y4) < 100):
#                pts4.append(pts[i])
#        
#        old_points1 = np.array(pts1, dtype = np.float32)
#        old_points2 = np.array(pts2, dtype = np.float32)
#        old_points3 = np.array(pts3, dtype = np.float32)
#        old_points4 = np.array(pts4, dtype = np.float32)
#
#
#    cv2.circle(frame, point1, 5, (0,0,255), 2)
#    cv2.circle(frame, point2, 5, (0,0,255), 2)
#    cv2.circle(frame, point3, 5, (0,0,255), 2)
#    cv2.circle(frame, point4, 5, (0,0,255), 2)
#
#    new_points1, status, error = cv2.calcOpticalFlowPyrLK(old_gray, gray_frame, old_points1, None, **lk_params)
#    new_points2, status, error = cv2.calcOpticalFlowPyrLK(old_gray, gray_frame, old_points2, None, **lk_params)
#    new_points3, status, error = cv2.calcOpticalFlowPyrLK(old_gray, gray_frame, old_points3, None, **lk_params)
#    new_points4, status, error = cv2.calcOpticalFlowPyrLK(old_gray, gray_frame, old_points4, None, **lk_params)
#
#    diff_x, diff_y, diff_x2, diff_y2, diff_x3, diff_y3, diff_x4, diff_y4 = [], [], [], [], [], [], [], []
#    
#    diffs = []
#    for i in range(0,len(new_points1)):
#        diff_x.append(new_points1[i][0] - old_points1[i][0])
#        diff_y.append(new_points1[i][1] - old_points1[i][1])
#        
#    for i in range(0,len(new_points2)):
#        diff_x2.append(new_points2[i][0] - old_points2[i][0])
#        diff_y2.append(new_points2[i][1] - old_points2[i][1])
#        
#    for i in range(0,len(new_points3)):
#        diff_x3.append(new_points3[i][0] - old_points3[i][0])
#        diff_y3.append(new_points3[i][1] - old_points3[i][1])
#        
#    for i in range(0,len(new_points4)):
#        diff_x4.append(new_points4[i][0] - old_points4[i][0])
#        diff_y4.append(new_points4[i][1] - old_points4[i][1])
#        
#    deltas = []
#    
#   
#    delta_x = statistics.median(diff_x)
#    delta_y = statistics.median(diff_y)
#    
#    delta_x2 = statistics.median(diff_x2)
#    delta_y2 = statistics.median(diff_y2)
#    
#    delta_x3 = statistics.median(diff_x3)
#    delta_y3 = statistics.median(diff_y3)
#    
#    delta_x4 = statistics.median(diff_x4)
#    delta_y4 = statistics.median(diff_y4)
# 
#    x = x + delta_x
#    y = y + delta_y
#    x = x.astype(np.float32)
#    y = y.astype(np.float32)
#    
#    x2 = x2 + delta_x2
#    y2 = y2 + delta_y2
#    x2 = x2.astype(np.float32)
#    y2 = y2.astype(np.float32)
#    
#    x3 = x3 + delta_x3
#    y3 = y3 + delta_y3
#    x3 = x3.astype(np.float32)
#    y3 = y3.astype(np.float32)
#    
#    x4 = x4 + delta_x4
#    y4 = y4 + delta_y4
#    x4 = x4.astype(np.float32)
#    y4 = y4.astype(np.float32)
#
#    cv2.circle(frame, (x,y), 5, (0,255,0), -1)
#    cv2.circle(frame, (x2, y2), 5, (0,255,0), -1)
#    cv2.circle(frame, (x3, y3), 5, (0,255,0), -1)
#    cv2.circle(frame, (x4, y4), 5, (0,255,0), -1)
#    
#
#    cv2.imshow("Frame", frame)
#    out.write(frame)
#    
#    count = count + 1
#    
#    old_gray = gray_frame.copy()
#    old_points1 = new_points1
#    old_points2 = new_points2
#    old_points3 = new_points3
#    old_points4 = new_points4
#
#    if cv2.waitKey(1) & 0xFF == ord('q'):
#        break

#One dimensional lists
#original_points = [193, 201, 731, 156, 1147, 532, 391, 627]
#
#points = [193, 201, 731, 156, 1147, 532, 391, 627]
#
#while True:
#    ret, frame = cap.read()
#    start_time = time.time()
#
#    
#    if ret:
#        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#    else:
#        print('Video has ended or failed, try a different video format!')
#        break
#    
#    if count % 10 == 0:
#        sift = cv2.xfeatures2d.SIFT_create()
#        kp = sift.detect(gray_frame,None)
#        pts = cv2.KeyPoint_convert(kp)
#        pts1, pts2, pts3, pts4 = [], [], [], []
#
#        for i in range(0,len(pts)):
#            if (abs(pts[i][0] - points[0]) < 100 and abs(pts[i][1] - points[1]) < 100):
#                pts1.append(pts[i])
#            if (abs(pts[i][0] - points[2]) < 100 and abs(pts[i][1] - points[3]) < 100):
#                pts2.append(pts[i])
#            if (abs(pts[i][0] - points[4]) < 100 and abs(pts[i][1] - points[5]) < 100):
#                pts3.append(pts[i])
#            if (abs(pts[i][0] - points[6]) < 100 and abs(pts[i][1] - points[7]) < 100):
#                pts4.append(pts[i])
#        
#        old_points1 = np.array(pts1, dtype = np.float32)
#        old_points2 = np.array(pts2, dtype = np.float32)
#        old_points3 = np.array(pts3, dtype = np.float32)
#        old_points4 = np.array(pts4, dtype = np.float32)
#
#    cv2.circle(frame, (original_points[0], original_points[1]), 5, (0,0,255), 2)
#    cv2.circle(frame, (original_points[2], original_points[3]), 5, (0,0,255), 2)
#    cv2.circle(frame, (original_points[4], original_points[5]), 5, (0,0,255), 2)
#    cv2.circle(frame, (original_points[6], original_points[7]), 5, (0,0,255), 2)
#
#    new_points1, status, error = cv2.calcOpticalFlowPyrLK(old_gray, gray_frame, old_points1, None, **lk_params)
#    new_points2, status, error = cv2.calcOpticalFlowPyrLK(old_gray, gray_frame, old_points2, None, **lk_params)
#    new_points3, status, error = cv2.calcOpticalFlowPyrLK(old_gray, gray_frame, old_points3, None, **lk_params)
#    new_points4, status, error = cv2.calcOpticalFlowPyrLK(old_gray, gray_frame, old_points4, None, **lk_params)
#    
#    diffs = [[],[],[],[],[],[],[],[]]
#    for i in range(0,len(new_points1)):
#        diffs[0].append(new_points1[i][0] - old_points1[i][0])
#        diffs[1].append(new_points1[i][1] - old_points1[i][1])
#        
#    for i in range(0,len(new_points2)):
#        diffs[2].append(new_points2[i][0] - old_points2[i][0])
#        diffs[3].append(new_points2[i][1] - old_points2[i][1])
#
#        
#    for i in range(0,len(new_points3)):
#        diffs[4].append(new_points3[i][0] - old_points3[i][0])
#        diffs[5].append(new_points3[i][1] - old_points3[i][1])
#
#    for i in range(0,len(new_points4)):
#        diffs[6].append(new_points4[i][0] - old_points4[i][0])
#        diffs[7].append(new_points4[i][1] - old_points4[i][1])
#              
#    deltas = []    
#    for i in range(0, len(diffs)):
#        deltas.append(statistics.median(diffs[i]))
# 
#    for i in range(0, len(points)):
#        points[i] = points[i] + deltas[i]
#        points[i] = points[i].astype(np.float32)
#    
#    cv2.circle(frame, (points[0], points[1]), 5, (0,255,0), -1)
#    cv2.circle(frame, (points[2], points[3]), 5, (0,255,0), -1)
#    cv2.circle(frame, (points[4], points[5]), 5, (0,255,0), -1)
#    cv2.circle(frame, (points[6], points[7]), 5, (0,255,0), -1)
#    
#    cv2.imshow("Frame", frame)
#    out.write(frame)
#    
#    count = count + 1
#    
#    old_gray = gray_frame.copy()
#    old_points1 = new_points1
#    old_points2 = new_points2
#    old_points3 = new_points3
#    old_points4 = new_points4
#    fps = 1.0 / (time.time() - start_time)
#    print("FPS: %.2f" % fps)
#
#    if cv2.waitKey(1) & 0xFF == ord('q'):
#        break