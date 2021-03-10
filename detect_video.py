    
import time
import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
from absl import app, flags, logging
from absl.flags import FLAGS
import core.utils as utils
from core.yolov4 import filter_boxes
from tensorflow.python.saved_model import tag_constants
from PIL import Image
import cv2
import numpy as np
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

from sortmaster.sort import *
import statistics

flags.DEFINE_string('framework', 'tf', '(tf, tflite, trt')
flags.DEFINE_string('weights', './checkpoints/yolov4-416',
                    'path to weights file')
flags.DEFINE_integer('size', 416, 'resize images to')
flags.DEFINE_boolean('tiny', False, 'yolo or yolo-tiny')
flags.DEFINE_string('model', 'yolov4', 'yolov3 or yolov4')
flags.DEFINE_string('video', './data/video/video.mp4', 'path to input video or set to 0 for webcam')
flags.DEFINE_string('output', None, 'path to output video')
flags.DEFINE_string('output_format', 'XVID', 'codec used in VideoWriter when saving video to file')
flags.DEFINE_float('iou', 0.25, 'iou threshold')
flags.DEFINE_float('score', 0.12, 'score threshold')
flags.DEFINE_boolean('dont_show', False, 'dont show video output')

#colors = [[0,0,0], [50,205,50], [60,20,220], [255,0,255], [255,255,0], [128,0,128], [255,0,0],[255,191,0], [0, 165,255], 
          #[255,255,255], [0,255,0], [0,255,255],[0,128,128]]

colors = [[0,0,0], [0,0,255], [255,255,255], [0,128,185],[0,225,0], [255, 0, 0], [255,255,51],
         [204,0,102], [100,100,100],[0, 255, 255], [255,0,255], [0,0,102], [135, 225, 0], [51, 255, 51], [117, 0, 220], 
         [153, 0, 153], [255,51, 153], [102, 51, 0], [75, 75, 75], [204, 102, 0], [206, 43, 72], [191, 0, 255], [255, 224, 102], 
         [168, 255, 187], [0, 76, 92], [56, 255, 167], [80, 255, 5], [92, 0, 49], [255, 148, 181], 
         [0, 194, 136], [241, 94, 242], [163, 240, 255], [204, 157, 0], [63, 153, 0], [170, 170, 170], [164, 255, 5], [124, 143, 0], [51, 0, 128], [204, 255, 153], [10, 116, 255]]
#colors = [[163, 240, 255],  , , , 
     #      [124, 143, 0], [204, 157, 0], , [0, 255, 16], [241, 94, 242],
     #      [255, 224, 102], ]
color_dict = {}

draw_boxes = False
object_tracking = True
siftLK = True
homography_no_tracking = False

#four_corners = [[263, 244], [918, 208], [1207, 485], [262, 564]]

#four_corners = [[209,238], [503, 214], [860, 287], [987, 401]]

#four_corners = [[481, 217], [919, 181], [1043, 286], [765, 244]]
#basketball_court = np.float32([[210,0], [441,0],[440, 182],[356, 75]])

#four_corners = np.array([[103, 242], [788,185], [1280, 659], [334, 720]], np.float32)
#basketball_court = np.array([[129,0], [513, 0], [513, 686], [129, 686]], np.float32)

#basketball_court = np.float32([[0,0], [331,0],[331, 630],[0, 630]])

#four_corners = [[263, 244], [918, 208], [1207, 485], [262, 564]]

#four_corners = [[429, 360], [927, 311], [102, 241], [531, 520]]
#basketball_court = np.float32([[105,219], [331,217], [0,0], [106, 416]])

#four_corners = [[501, 244], [981, 285], [766, 662], [87, 581]]
#basketball_court = np.float32([[0,0], [335, 0], [335, 630], [0,630]])

#four_corners = [[395, 331], [743, 368], [657, 479], [270, 435]]
#basketball_court = np.float32([[0,214], [229, 214], [229, 414], [0,414]])
#basketball_court = np.float32([[0, 232], [266, 232], [266,455], [0,455]])

#four_corners = [[426, 314], [981, 341], [151, 512], [966, 572]]
#basketball_court = np.float32([[0,0], [480,0], [480, 511], [0, 511]])

four_corners = np.array([[501, 244], [981, 285], [766, 662], [87, 581]], np.float32)

court = cv2.imread("./courts/celtics3.png")

output_width = court.shape[1]
output_height = court.shape[0] 
basketball_court = np.array([[0,0], [384, 0], [384, output_height], [0, output_height]], np.float32)


#Smart points [[103, 242], [788,185], [1280, 659], [334, 720]]
#Bojan points [[580, 242], [1186, 289], [1083, 645], [238,508]]
#Lee points [[871, 380], [1473, 424], [1238, 878], [17, 760]]
#Smart 2 [[263, 244], [918, 208], [1207, 485], [262, 564]]
#Giannis [[426, 314], [981, 341], [151, 512], [966, 572]]
#Kemba [[501, 244], [981, 285], [766, 662], [87, 581]]
#PR2 (does not work as well) points = [[299, 272], [812, 239], [1277, 547], [537, 620]], [[299, 272], [812, 239], [1111, 436], [649, 450]]
#PR1 [[405, 240], [1028, 271], [918, 719], [23, 654]]
#Horford [[193, 201], [731, 156], [1147, 532], [391, 627]]




# The output width:x
    #Courtney Lee: output_width = 507
    #Kemba: output_width = 515
    #Sexton: output_width = 370
    #Smart1: output_width = 386
    #Smart2: output_width = 560
    #Horford: output_width = 331
    #Bojan: output_width = 390
    #No Homography: output_width = 1280
    #Pick and Roll 1: output_width = 406
    #Pick and Roll 2: output_width = 583
    #Giannis: 480



# List of the output heights for each of the videos
    #Courtney Lee: output_height = 903
    #Kemba: output_height = 687
    #Sexton: output_height = 542
    #Smart1: output_height = 687
    #Smart2: output_height = 687
    #Horford: output_height = 631
    #No Homography: output_height = 720
    #Bojan: output_height = 596
    #PickandRoll1: output_height = 671
    #PickandRoll2: output_height = 807
    #Giannis: 511


if draw_boxes == True:
    output_width = 1280
    output_height = 720

# The homography width and height are equal to the output width and height in most cases
homography_height = output_height
homography_width = output_width

mot_tracker = Sort() 

def main(_argv):
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)
    STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config(FLAGS)
    input_size = FLAGS.size
    video_path = FLAGS.video


    if FLAGS.framework == 'tflite':
        interpreter = tf.lite.Interpreter(model_path=FLAGS.weights)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        print(input_details)
        print(output_details)
    else:
        saved_model_loaded = tf.saved_model.load(FLAGS.weights, tags=[tag_constants.SERVING])
        infer = saved_model_loaded.signatures['serving_default']

    # begin video capture
    try:
        vid = cv2.VideoCapture(int(video_path))
    except:
        vid = cv2.VideoCapture(video_path)


    out = None
    
    if FLAGS.output:
        # by default VideoCapture returns float instead of int
        #width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        width = output_width
        #height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        height = output_height
        fps = int(vid.get(cv2.CAP_PROP_FPS))
        codec = cv2.VideoWriter_fourcc(*FLAGS.output_format)
        out = cv2.VideoWriter(FLAGS.output, codec, fps, (width, height))

    if siftLK == True:
        _, frame = vid.read()
        old_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        lk_params = dict(winSize = (10,10), 
                     maxLevel = 2, 
                     criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        count = 0
    
    while True:
        return_value, frame = vid.read()
        if return_value:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame)
        else:
            print(color_dict)
            print('Video has ended or failed, try a different video format!')
            break
    
        frame_size = frame.shape[:2]
        image_data = cv2.resize(frame, (input_size, input_size))
        image_data = image_data / 255.
        image_data = image_data[np.newaxis, ...].astype(np.float32)
        start_time = time.time()

        if FLAGS.framework == 'tflite':
            interpreter.set_tensor(input_details[0]['index'], image_data)
            interpreter.invoke()
            pred = [interpreter.get_tensor(output_details[i]['index']) for i in range(len(output_details))]
            if FLAGS.model == 'yolov3' and FLAGS.tiny == True:
                boxes, pred_conf = filter_boxes(pred[1], pred[0], score_threshold=0.25,
                                                input_shape=tf.constant([input_size, input_size]))
            else:
                boxes, pred_conf = filter_boxes(pred[0], pred[1], score_threshold=0.25,
                                                input_shape=tf.constant([input_size, input_size]))
        else:
            batch_data = tf.constant(image_data)
            pred_bbox = infer(batch_data)
            for key, value in pred_bbox.items():
                boxes = value[:, :, 0:4]
                pred_conf = value[:, :, 4:]

        boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
            boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
            scores=tf.reshape(
                pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
            max_output_size_per_class=50,
            max_total_size=50,
            iou_threshold=FLAGS.iou,
            score_threshold=FLAGS.score
        )
        pred_bbox = [boxes.numpy(), scores.numpy(), classes.numpy(), valid_detections.numpy()]
        
        #Remove all of the objects that are not people
        player_ixs = np.where(pred_bbox[2][0] == 0)[0]
        pred_bbox[0] = pred_bbox[0][:,player_ixs]
        pred_bbox[1] = pred_bbox[1][:,player_ixs]
        pred_bbox[2] = pred_bbox[2][:,player_ixs]
              
        if siftLK == True:
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if count % 10 == 0:
                sift = cv2.xfeatures2d.SIFT_create()
                kp = sift.detect(gray_frame,None)
                pts = cv2.KeyPoint_convert(kp)
                pts1, pts2, pts3, pts4 = [], [], [], []
                for i in range(0,len(pts)):
                    if (abs(pts[i][0] - four_corners[0][0]) < 150 and abs(pts[i][1] - four_corners[0][1]) < 150):
                        pts1.append(pts[i])
                    if (abs(pts[i][0] - four_corners[1][0]) < 150 and abs(pts[i][1] - four_corners[1][1]) < 150):
                        pts2.append(pts[i])
                    if (abs(pts[i][0] - four_corners[2][0]) < 150 and abs(pts[i][1] - four_corners[2][1]) < 150):
                        pts3.append(pts[i])
                    if (abs(pts[i][0] - four_corners[3][0]) < 150 and abs(pts[i][1] - four_corners[3][1]) < 150):
                        pts4.append(pts[i])
                
                old_points1 = np.array(pts1, dtype = np.float32)
                old_points2 = np.array(pts2, dtype = np.float32)
                old_points3 = np.array(pts3, dtype = np.float32)
                old_points4 = np.array(pts4, dtype = np.float32)
                
        
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
                      
            deltas = [[],[],[],[]]
            
            for i in range(0, len(diffs)):
                for j in range(0, len(diffs[i])):
                    deltas[i].append(statistics.median(diffs[i][j]))
                    
            for i in range(0, len(four_corners)):
                for j in range(0, len(four_corners[i])):      
                    four_corners[i][j] = four_corners[i][j] + deltas[i][j]
                    four_corners[i][j] = four_corners[i][j].astype(np.float32)
                
            old_gray = gray_frame.copy()
            old_points1 = new_points1
            old_points2 = new_points2
            old_points3 = new_points3
            old_points4 = new_points4
            
            fourcorners = np.array(four_corners)
            count = count + 1
    
        #Use these three lines when not using object tracking:
        if homography_no_tracking == True:
            points = get_points(pred_bbox[0], frame)
            homography_points = homographypts(points)
            image = draw_pts(homography_points,court)
       
        #Use these two lines when drawing bounding boxes
        if draw_boxes == True:
            points = get_points(pred_bbox[0], frame)
            image = utils.draw_bbox(frame, pred_bbox)
            draw_pts(points, image)

        #Uses the object tracking algorithm
        if object_tracking == True:
            detections = get_tracking_points(pred_bbox, frame)
            track_bbs_ids = mot_tracker.update(detections)
            trackedpts = homographypts_tracked(track_bbs_ids)
            image = draw_tracked_pts(trackedpts,court)
            
        result = np.asarray(image)
        
        cv2.namedWindow("result", cv2.WINDOW_AUTOSIZE)
        if draw_boxes == True:
            result = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        fps = 1.0 / (time.time() - start_time)
        print(count, "FPS: %.2f" % fps)
        
        if not FLAGS.dont_show:
            cv2.imshow("result", result)
        if FLAGS.output:
            out.write(result)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
    cv2.destroyAllWindows()

# A helper method 
def get_points(boxes,img):
    count = len(boxes[0])
    points = []
    width = img.shape[1]
    height = img.shape[0]
    for x in range(0, count):
        point = []
        bottom = int(boxes[0][x][2] * height)
        middle = int((boxes[0][x][1] * width +  boxes[0][x][3] * width) / 2)
        point = [middle, bottom]
        points.append(point)
    return points

# get_tracking_points - Takes the output of the YOLOv4 detection and converts 
# it to the format required for the sort algorithm. The required format is
# x1, y1, x2, y2, score. The output is a list of all of the points in that format
    
def get_tracking_points(objects, img):
    points = []
    width = img.shape[1]
    height = img.shape[0]
    for i in range(0, len(objects[0][0])):
        new_point = [0,0,0,0,0]
        new_point[0] = int(objects[0][0][i][1] * width)
        new_point[1] = int(objects[0][0][i][0] * height)
        new_point[2] = int(objects[0][0][i][3] * width)
        new_point[3] = int(objects[0][0][i][2] * height) 
        new_point[4] = objects[1][0][i]
        points.append(new_point)
    return points

# draw_pts - Draw the points on the image without object tracking. All of the points are black
def draw_pts(points, img):
    for x in range(0, len(points)):
        cv2.circle(img, ((points[x][0]),(points[x][1])), radius=2, color=(0, 0, 0), thickness=2)
    return img


#homographypts - this function computes the homography without object tracking
# and removes the points outside the relevant area.
def homographypts(points):

    #basketball_court = np.float32([[0,0],[homography_width, 0], [homography_width,homography_height], [0, homography_height]])

    matrix = cv2.getPerspectiveTransform(four_corners,basketball_court)
    points = np.asarray(points).reshape(-1, 1, 2).astype(np.float64)
    transformed_pts = cv2.perspectiveTransform(points,matrix) 
    transformed_pts = transformed_pts.astype(int).reshape(-1, 2)  
    
    #Remove all points outside the relevant area
    final_pts = transformed_pts[(transformed_pts[:,0] > 0) & (transformed_pts[:,0] < output_width) &\
                (transformed_pts[:,1] > 0) & (transformed_pts[:,1] < output_height)]
    return final_pts


def homographypts_tracked(boxes):
    #Converts the array to the format with the bottom middle of the boxes 
    points=[]
    scores = []
    for i in range(0, len(boxes)):
        point = []
        # Make sure the points are not nan
        if boxes[i][0] >= 0:
            bottom = int(boxes[i][3])
            score = int(boxes[i][4])
            middle = int((boxes[i][0] + boxes[i][2])/2)
            point = [middle, bottom]
            points.append(point)
            scores.append(score)
    
    #basketball_court = np.float32([[0,0],[homography_width, 0], [homography_width,homography_height], [0, homography_height]])
    #Compute the homography and convert the points
    matrix = cv2.getPerspectiveTransform(four_corners, basketball_court)
    points = np.asarray(points).reshape(-1, 1, 2).astype(np.float64)
    transformed_pts = cv2.perspectiveTransform(points, matrix)    
    transformed_pts = transformed_pts.astype(int).reshape(-1, 2)
    
    #Add the scores to the list of tracked points
    new_pts = []
    for i in range(0, len(transformed_pts)):
        new_point = [transformed_pts[i][0], transformed_pts[i][1], scores[i]]
        new_pts.append(new_point) 
    new_pts = np.array(new_pts)

    #Remove all points outside the relevant area
    final_pts = new_pts[(new_pts[:,0] > 0) & (new_pts[:,0] < output_width) & (new_pts[:,1] > 0) & (new_pts[:,1] < output_height)]
    return final_pts

#draw_tracked_pts - Draws the points on the image with object tracking. 
#The colors of the points are determined by the object tracking.
def draw_tracked_pts(points, img): 
    for i in range(0, len(points)):
        #blue = int((20 * points[i][2] + 50)% 255)
        #green = int((30 * points[i][2] + 100)% 255)
        #red = int((40 * points[i][2] + 150)% 255)
        #(colors[index][0], colors[index][1], colors[index][2])
        #index = points[i][2] % len(colors), (colors[index][0], colors[index][1], colors[index][2])
        color = get_color(points[i][2])
        cv2.circle(img, ((points[i][0]),(points[i][1])), radius=2, color=(int(color[2]), int(color[0]), int(color[1])), thickness=2)
    return img


def get_color(tracking_id):
    if tracking_id in color_dict:
        return color_dict[tracking_id]
    else:
        color_dict[tracking_id] = colors[len(color_dict)]
        return color_dict[tracking_id]
#Helper methods that are no longer used

#def draw_transformed_pts(points, img):
#    for i in range(0, len(points)):
#        cv2.circle(img, ((points[i][0][0]),(points[i][0][1])), radius=3, color=(0, 0, 0), thickness=3)
#    return img

#def homographyimg(img):
#    photopts = np.float32([[871, 381],[1470, 429],[1238, 877],[319,758]])
#    basketball_court = np.float32([[0,0],[output_width, 0], [output_width,output_height], [0,output_height]])
#    matrix = cv2.getPerspectiveTransform(photopts,basketball_court)
#    new = cv2.warpPerspective(img, matrix, (output_width, output_height)
#    return new
# List of the four corners for each of the videos
    #CourtneyLeePoints = np.float32([[871, 381],[1472, 424],[1238, 877],[319,758]])
    #Kembapts = np.float32([[501, 244],[1160, 298],[1057, 703],[89,579]])
    #GiannisPts = np.float32([[427, 314],[1034, 345],[1020, 572],[153,512]])
    #Sexton2Pts = np.float32([[0, 335],[596, 147],[1280, 500],[841,697]])
    #Sexton1Pts = np.float32([[301, 261],[965, 278],[887, 623],[276,574]])
    #HorfordPts = np.float32([[189, 199],[727, 153],[1143, 529],[378,624]])
    #Smart1Pts = np.float32([[240, 254],[798, 209],[1247, 622],[434,701]])
    #Smart2Pts = np.float32([[225, 247],[900, 208],[1217, 509],[225,558]])
    #Bojan_Points = np.float32([[581, 242],[1101, 282],[987, 632],[232,507]])
    #PickandRoll2 = np.float32([[86,282],[720,238],[1186,544],[223,670] ])
    #PickandRoll1 = np.float32([[424,246],[1125,283],[1013,720],[50,654] ])
    #Sexton2Pts = np.float32([[0, 332],[575, 154],[1280, 500],[852,691]])
   

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
