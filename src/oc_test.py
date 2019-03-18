"""Testing openCV and yolo-v3
original code in ivangrov
        YOLOv3-Series/[part 1]YOLOv3_with_OpenCV/OD.py
"""

import numpy as np
import cv2 as cv
import time

VIDEO_FILE = "data/hockey_test.mp4"

CONF_THRESHOLD = 0.25
NMS_THRESHOLD = 0.40
INP_WIDTH = 416
INP_HEIGHT = 416

# Model configuration
MODEL_CONF = 'src/darkflow/cfg/yolov3.cfg'
MODEL_WEIGHTS = 'src/darkflow/bin/yolov3-608.weights'


# Load names of classes and turn that into a list
CLASSES_FILE = "src/darkflow/cfg/coco.names"
classes = None

with open(CLASSES_FILE, 'rt') as f:
    classes = f.read().rstrip('\n').split('\n')


def postprocess(frame, outs):
    """postprocess description"""

    frame_height = frame.shape[0]
    frame_width = frame.shape[1]

    class_ids = []
    confidences = []
    boxes = []

    for out in outs:
        for detection in out:

            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > CONF_THRESHOLD:
                center_x = int(detection[0] * frame_width)
                center_y = int(detection[1] * frame_height)

                width = int(detection[2] * frame_width)
                height = int(detection[3]*frame_height)

                left = int(center_x - width/2)
                top = int(center_y - height/2)

                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([left, top, width, height])

    indices = cv.dnn.NMSBoxes(boxes, confidences, CONF_THRESHOLD, NMS_THRESHOLD)

#    indices = cv.dnn.NMSBoxes(boxes, confidences, CONF_THRESHOLD, NMS_THRESHOLD)
    for i in indices:
        i = i[0]
        box = boxes[i]
        left = box[0]
        top = box[1]
        width = box[2]
        height = box[3]
        if classes[class_ids[i]] == 'person':
            drawPred(class_ids[i], confidences[i], left, top, left + width, top + height)


def drawPred(class_id, conf, left, top, right, bottom):
    #    print("label: {}\tconfidence: {:.3f}\ttl:({},{})\tbr:({},{})".format(
    #        classes[class_id], conf, left, top, right, bottom)
    #    )
    # Draw a bounding box.
    cv.rectangle(frame, (left, top), (right, bottom), (255, 178, 50), 3)

    label = '%.2f' % conf

    # Get the label for the class name and its confidence
    if classes:
        assert (class_id < len(classes))
        label = '%s:%s' % (classes[class_id], label)

    # A fancier display of the label from learnopencv.com
    # Display the label at the top of the bounding box
    # labelSize, baseLine = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    # top = max(top, labelSize[1])
    # cv.rectangle(frame, (left, top - round(1.5 * labelSize[1])), (left + round(1.5 * labelSize[0]), top + baseLine),
    # (255, 255, 255), cv.FILLED)
    # cv.rectangle(frame, (left,top),(right,bottom), (255,255,255), 1 )
    # cv.putText(frame, label, (left, top), cv.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 1)
    cv.putText(frame, label, (left, top), cv.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)


def getOutputsNames(net):
    # Get the names of all the layers in the network
    layers_names = net.getLayerNames()

    # Get the names of the output layers, i.e. the layers with unconnected outputs
    return [layers_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]


# Set up the net

net = cv.dnn.readNetFromDarknet(MODEL_CONF, MODEL_WEIGHTS)
net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)


# Process inputs
win_name = 'DL OD with OpenCV'
cv.namedWindow(win_name, cv.WINDOW_NORMAL)
cv.resizeWindow(win_name, 1000, 1000)


# cap = cv.VideoCapture(0)
cap = cv.VideoCapture(VIDEO_FILE)

while cv.waitKey(1) < 0:
    stime = time.time()

    # get frame from video
    has_frame, frame = cap.read()

    # Create a 4D blob from a frame
    if has_frame:
        blob = cv.dnn.blobFromImage(frame, 1/255, (INP_WIDTH, INP_HEIGHT), [0, 0, 0], 1, crop=False)

        # Set the input the the net
        net.setInput(blob)
        outs = net.forward(getOutputsNames(net))

        postprocess(frame, outs)

        # show the image
        cv.imshow(win_name, frame)

        # print('FPS {:.1f}'.format(1/(time.time() - stime)))

    else:
        cap.release()
        break
