#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 14 11:57:41 2021

@author: jonasrajagopal
"""
import cv2
import numpy as np

cap = cv2.VideoCapture("Smart2ndFoul.mp4")
sift = cv2.xfeatures2d.SIFT_create()

img = cv2.imread('Smart1.jpg')
gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
sift = cv2.xfeatures2d.SIFT_create()
kp = sift.detect(gray,None)
img=cv2.drawKeypoints(gray,kp,img)
pts = cv2.KeyPoint_convert(kp)
cv2.imwrite('sift_keypoints2.jpg',img)


for i in range(0,len(pts)):
    pts[i][0] = round(pts[i][0])
    pts[i][1] = round(pts[i][1])

pts= pts.astype(int)
print(pts)
#pts = np.float([kp[idx].pt for idx in range(0, len(kp))]).reshape(-1, 1, 2)

#For a video:
#while True:
#    _, frame = cap.read()
#    gray= cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
#    kp = sift.detect(gray,None)
#    frame=cv2.drawKeypoints(gray,kp,frame)
#    cv2.imshow("Frame", frame)
#    key = cv2.waitKey(1)
#    if key == 27:
#        break


cap.release()
cv2.destroyAllWindows()
