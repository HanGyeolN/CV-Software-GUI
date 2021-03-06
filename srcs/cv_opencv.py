import cv2
import numpy as np
import random
import os

HAAR_PATH = "./haar/haarcascade_frontalface_alt.xml"
MOBILENET_TXT = "./mobile-net/MobileNetSSD_deploy.prototxt.txt"
MOBILENET_PATH = "./mobile-net/MobileNetSSD_deploy.caffemodel"
MASK_RCNN_LABEL_PATH = "./mask-rcnn/object_detection_classes_coco.txt"
MASK_RCNN_COLOR_PATH = "./mask-rcnn/colors.txt"
MASK_RCNN_WEIGHTS_PATH = "./mask-rcnn/frozen_inference_graph.pb"
MASK_RCNN_CONFIG_PATH = "./mask-rcnn/mask_rcnn_inception_v2_coco_2018_01_28.pbtxt"

def equalize_histogram(self):
    """
    Histogram Equalization.
    """
    channel, nRow, nCol = self.input_array.shape
    self.output_array = np.zeros((channel, nRow, nCol))
    temp = self.input_array.astype(np.uint8)
    
    for RGB in range(channel):
        self.output_array[RGB] = cv2.equalizeHist(temp[RGB])

def face_detection(self):
    """
    Detect face from image using haar.
    """
    face_cascade = cv2.CascadeClassifier(HAAR_PATH)
    cvPhoto2 = self.input_array.transpose(1, 2, 0).copy()
    cvPhoto2 = cvPhoto2.astype(np.uint8)
    grey = cv2.cvtColor(cvPhoto2, cv2.COLOR_RGB2GRAY)
    face_rects = face_cascade.detectMultiScale(grey, 1.1, 5)

    for (x, y, w, h) in face_rects:
        cv2.rectangle(cvPhoto2, (x, y), (x + w, y + h), (0, 255, 0), 3)
    self.output_array = cvPhoto2.transpose(2, 0, 1)
    self.setStatusTip(f"사람수: {len(face_rects)} 위치: {face_rects}")

def object_detection(self):
    """
    Object detection using Mobile Net.
    """
    CONF_VALUE = 0.2
    CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
                "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
                "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
                "sofa", "train", "tvmonitor"]
    COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))
    net = cv2.dnn.readNetFromCaffe(MOBILENET_TXT, MOBILENET_PATH)
    image = self.input_array.transpose(1,2,0).copy()
    image = image.astype(np.uint8)
    (h, w) = image.shape[:2]
    photo = image.copy()
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 
                                 0.007843, 
                                 (300, 300), 
                                 127.5)
    net.setInput(blob)
    detections = net.forward()
    # loop over the detections
    for i in np.arange(0, detections.shape[2]): # 여러개를 찾기위한 for문
        confidence = detections[0, 0, i, 2]
        if confidence > CONF_VALUE:
            idx = int(detections[0, 0, i, 1]) # 네모 박스
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            # 예측 결과, 확률 표시
            label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)
            print("[INFO] {}".format(label))
            cv2.rectangle(photo, (startX, startY), (endX, endY), COLORS[idx], 4)
            y = startY - 15 if startY - 15 > 15 else startY + 15
            cv2.putText(photo, label, (startX, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, COLORS[idx], 2)
    self.output_array = photo.transpose(2,0,1)

def object_detection_mask_rcnn(self):
    CONF_VALUE = 0.5
    THRESHOLD_VALUE = 0.3
    VISUALIZE = 0
    LABELS = open(MASK_RCNN_LABEL_PATH).read().strip().split("\n")

    # load the set of colors that will be used when visualizing a given
    # instance segmentation
    COLORS = open(MASK_RCNN_COLOR_PATH).read().strip().split("\n")
    COLORS = [np.array(c.split(",")).astype("int") for c in COLORS]
    COLORS = np.array(COLORS, dtype="uint8")

    # load our Mask R-CNN trained on the COCO dataset (90 classes)
    # from disk
    print("[INFO] loading Mask R-CNN from disk...")
    net = cv2.dnn.readNetFromTensorflow(
        MASK_RCNN_WEIGHTS_PATH, 
        MASK_RCNN_CONFIG_PATH
    )

    # load our input image and grab its spatial dimensions
    image = self.input_array.transpose(1,2,0).copy()
    image = image.astype(np.uint8)
    (H, W) = image.shape[:2]

    # construct a blob from the input image and then perform a forward
    # pass of the Mask R-CNN, giving us (1) the bounding box  coordinates
    # of the objects in the image along with (2) the pixel-wise segmentation
    # for each specific object

    blob = cv2.dnn.blobFromImage(image, swapRB=True, crop=False)
    net.setInput(blob)
    (boxes, masks) = net.forward(["detection_out_final", "detection_masks"])

    # show timing information and volume information on Mask R-CNN
    print("[INFO] boxes shape: {}".format(boxes.shape))
    print("[INFO] masks shape: {}".format(masks.shape))

    # loop over the number of detected objects
    clone = image.copy()
    for i in range(0, boxes.shape[2]):
        # extract the class ID of the detection along with the confidence
        # (i.e., probability) associated with the prediction
        classID = int(boxes[0, 0, i, 1])
        confidence = boxes[0, 0, i, 2]

        # filter out weak predictions by ensuring the detected probability
        # is greater than the minimum probability
        if confidence > CONF_VALUE:
            # clone our original image so we can draw on it


            # scale the bounding box coordinates back relative to the
            # size of the image and then compute the width and the height
            # of the bounding box
            box = boxes[0, 0, i, 3:7] * np.array([W, H, W, H])
            (startX, startY, endX, endY) = box.astype("int")
            boxW = endX - startX
            boxH = endY - startY

            # extract the pixel-wise segmentation for the object, resize
            # the mask such that it's the same dimensions of the bounding
            # box, and then finally threshold to create a *binary* mask
            mask = masks[i, classID]
            mask = cv2.resize(mask, (boxW, boxH),
                                interpolation=cv2.INTER_NEAREST)
            mask = (mask > THRESHOLD_VALUE)

            # extract the ROI of the image
            roi = clone[startY:endY, startX:endX]

            # check to see if are going to visualize how to extract the
            # masked region itself
            if VISUALIZE > 0:
                # convert the mask from a boolean to an integer mask with
                # to values: 0 or 255, then apply the mask
                visMask = (mask * 255).astype("uint8")
                instance = cv2.bitwise_and(roi, roi, mask=visMask)


            # now, extract *only* the masked region of the ROI by passing
            # in the boolean mask array as our slice condition
            roi = roi[mask]

            # randomly select a color that will be used to visualize this
            # particular instance segmentation then create a transparent
            # overlay by blending the randomly selected color with the ROI
            color = random.choice(COLORS)
            blended = ((0.4 * color) + (0.6 * roi)).astype("uint8")

            # store the blended ROI in the original image
            clone[startY:endY, startX:endX][mask] = blended

            # draw the bounding box of the instance on the image
            color = [int(c) for c in color]

            # draw the predicted label and associated probability of the
            # instance segmentation on the image
            text = "{}: {:.4f}".format(LABELS[classID], confidence)
            cv2.putText(clone, text, (startX, startY - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
    self.output_array = clone.transpose(2, 0, 1)