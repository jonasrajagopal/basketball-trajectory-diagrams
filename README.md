The YOLOv4 code is from:

TheAIGuysCode, & Nickvaras. (2020). TheAIGuysCode/yolov4-deepsort. Retrieved December 13, 2020, from https://github.com/theAIGuysCode/yolov4-deepsort

Complete instructions for how to run YOLOv4 can be found in the "YOLOv4 README.md" file. 

The SORT code is from:

Bewley, A., Ge, Z., Ott, L., Ramos, F., & Upcroft, B. (2016). Simple online and realtime tracking. 2016 IEEE International Conference on Image Processing (ICIP). doi:10.1109/icip.2016.7533003 and the code can be found at:
https://github.com/abewley/sort.

Details about SORT can be found in the sortmaster folder in "SORT README.md"

The SIFTLK method is incorporated into the detectvideo.py function, so the entire system can be run with detect_video.py.

There are four different settings at the top of the detect_video.py file. The typical settings are True for SIFTLK and Object_tracking and False for draw_boxes and homography_no_tracking. Homography_no_tracking produces only black trajectories and draw_boxes produces the result of the YOLOv4 object detection without any homography. When SIFTLK == True, the system in incorporates a correction for camera motion and when Object_Tracking == True, the system incorporates the SORT object tracking algoritm and produces colored trajectories.

The thesis can be found at: 
http://bit.ly/RajagopalSeniorThesis
