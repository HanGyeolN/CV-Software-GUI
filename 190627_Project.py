import sys
from PyQt5.QtWidgets import *
from PyQt5 import uic
from PyQt5.QtGui import *
import numpy as np
import PIL
import time
import sqlite3
import matplotlib.pyplot as plt
import math
import cv2
from scipy import ndimage
import matplotlib
import os
import random

from srcs import cv_files
from srcs import cv_image
from srcs import cv_basic

# pip install pillow numpy PyQt5 
# Anaconda/Library/bin/qt_designer.exe
# ui파일 저장경로

UIPath = "C:/Users/ssna/Desktop/PyQt2/qt_design_test.ui"
HAAR_PATH = "C:/Users/ssna/Desktop/PyQt2/haar/"
DB_PATH = "C:/Users/ssna/Desktop/PyQt2/PyQtCVImageDB.sqlite3"

form_class = uic.loadUiType(UIPath)[0]

class MyWindow(QMainWindow, form_class):    
    input_pixmap = None # QPixmap Object Image
    input_image = None # QImage Object (0~1 사이 값)
    input_array = None # npArray Image (0~255 사이 값)

    output_pixmap = None
    output_image = None
    output_array = None

    viewRow = 500
    viewCol = 500

    def __init__(self):
        super().__init__()
        self.setupUi(self)
        ### 파일처리
        self.Out2InButton.clicked.connect(self.out2in)
        self.actionOpenImage.triggered.connect(self.openImage)
        self.actionSaveImage.triggered.connect(self.saveImage)
        self.actionLoadFromDB.triggered.connect(self.loadFromDB)
        self.actionSaveToDB.triggered.connect(self.saveToDB)
        ### 화소
        self.actionBrightnessControl.triggered.connect(self.brightnessControl)
        self.actionBinary.triggered.connect(self.binary)
        ### 화소영역
        self.actionMasking.triggered.connect(self.masking)
        ### 기하
        self.actionResizing.triggered.connect(self.resizing)
        self.actionRotation.triggered.connect(self.rotation)
        ### 통계
        self.actionHistogram.triggered.connect(self.histogram)
        self.actionHistogramEqualization.triggered.connect(self.histogramEqualization)
        ### ML
        self.actionHaarFaceDetection.triggered.connect(self.haarFaceDetection)
        ### OpenCV
        self.actionObjectDetection.triggered.connect(self.objectDetection)
        self.actionMaskRCNN.triggered.connect(self.maskRCNN)

####### 파일처리
    def loadFromDB(self):
        cv_files.open_image_sqlite(self)

    def saveToDB(self):
        cv_files.save_image_sqlite(self)

    def openImage(self):
        cv_files.open_image_file(self)

    def saveImage(self):
        cv_files.save_image_file(self)  

    def out2in(self):
        cv_image.set_base_image(self)

    def displayOutputImage(self, nRow=0, nCol=0):
        cv_image.display_output_image(self, nRow, nCol)

####### 화소처리
    def brightnessControl(self):
        cv_basic.brightness_control(self)
        self.displayOutputImage()

    def binary(self):
        cv_basic.binarize_image(self)
        self.displayOutputImage()

    def masking(self):
        cv_basic.masking_image(self)
        self.displayOutputImage()

####### 기하처리
    def resizing(self):
        cv_basic.resize_image(self)
        self.displayOutputImage()       
    
    def rotation(self):
        cv_basic.rotate_image(self)
        self.displayOutputImage()

####### 통계처리
    def histogram(self):
        cv_basic.hist_image(self)

    def histogramEqualization(self):
        start = time.time()

        channel, nRow, nCol = self.input_array.shape

        self.output_array = np.zeros((channel, nRow, nCol))


        temp = self.input_array.astype(np.uint8)
        for RGB in range(channel):
            self.output_array[RGB] = cv2.equalizeHist(temp[RGB])
   
        self.displayOutputImage(nRow, nCol)



###### OpenCV
    def haarFaceDetection(self):
        start = time.time()
        face_cascade = cv2.CascadeClassifier(self.HAAR_PATH+"haarcascade_frontalface_alt.xml")

        cvPhoto2 = self.input_array.transpose(1, 2, 0).copy()
        cvPhoto2 = cvPhoto2.astype(np.uint8)

        grey = cv2.cvtColor(cvPhoto2, cv2.COLOR_RGB2GRAY)
        face_rects = face_cascade.detectMultiScale(grey, 1.1, 5)

        for (x, y, w, h) in face_rects:
            cv2.rectangle(cvPhoto2, (x, y), (x + w, y + h), (0, 255, 0), 3)

        self.output_array = cvPhoto2.transpose(2, 0, 1)
        self.displayOutputImage()
        self.setStatusTip("사람수: " + str(len(face_rects)) + " 위치: " + str(face_rects))

    def objectDetection(self):
        CONF_VALUE = 0.2  # 조절해야할 수치.

        CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
                   "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
                   "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
                   "sofa", "train", "tvmonitor"]

        COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

        net = cv2.dnn.readNetFromCaffe("C:/Users/ssna/Desktop/PyQt2/MobileNetSSD_deploy.prototxt.txt",
                                       "C:/Users/ssna/Desktop/PyQt2/MobileNetSSD_deploy.caffemodel")  ## opencv에서 지원
        # arg에 txt와 model파일 경로를 준다.
        # 이거때문에 opencv를 사용.

        image = self.input_array.transpose(1,2,0).copy()
        image = image.astype(np.uint8)

        (h, w) = image.shape[:2]

        photo = image.copy()
        blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 0.007843, (300, 300), 127.5)

        net.setInput(blob)
        detections = net.forward()

        # loop over the detections
        for i in np.arange(0, detections.shape[2]):  # 여러개를 찾기위한 for문
            confidence = detections[0, 0, i, 2]  # confidence는
            if confidence > CONF_VALUE:  # 수정사항.
                idx = int(detections[0, 0, i, 1])  # 네모 박스
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                # 예측 결과를 보여준다.
                label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)  # 몇퍼센트 확률로 무엇이다.
                print("[INFO] {}".format(label))
                cv2.rectangle(photo, (startX, startY), (endX, endY), COLORS[idx], 4)
                y = startY - 15 if startY - 15 > 15 else startY + 15
                cv2.putText(photo, label, (startX, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)

        # # show the output image
        # cv2.imshow("Output", image)  #
        self.output_array = photo.transpose(2,0,1)
        self.displayOutputImage()

    def maskRCNN(self):
        MASK_RCNN_PATH = "C:/Users/ssna/Desktop/PyQt2/mask-rcnn/mask-rcnn-coco"
        CONF_VALUE = 0.5
        THRESHOLD_VALUE = 0.3
        VISUALIZE = 0


        labelsPath = "C:/Users/ssna/Desktop/PyQt2/mask-rcnn/mask-rcnn-coco/object_detection_classes_coco.txt"
        LABELS = open(labelsPath).read().strip().split("\n")

        # load the set of colors that will be used when visualizing a given
        # instance segmentation
        colorsPath = MASK_RCNN_PATH+"/colors.txt"
        COLORS = open(colorsPath).read().strip().split("\n")
        COLORS = [np.array(c.split(",")).astype("int") for c in COLORS]
        COLORS = np.array(COLORS, dtype="uint8")

        # derive the paths to the Mask R-CNN weights and model configuration
        weightsPath = "C:/Users/ssna/Desktop/PyQt2/mask-rcnn/mask-rcnn-coco/frozen_inference_graph.pb"
        configPath = "C:/Users/ssna/Desktop/PyQt2/mask-rcnn/mask-rcnn-coco/mask_rcnn_inception_v2_coco_2018_01_28.pbtxt"

        # load our Mask R-CNN trained on the COCO dataset (90 classes)
        # from disk
        print("[INFO] loading Mask R-CNN from disk...")
        net = cv2.dnn.readNetFromTensorflow(weightsPath, configPath)

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
        start = time.time()
        (boxes, masks) = net.forward(["detection_out_final", "detection_masks"])
        end = time.time()

        # show timing information and volume information on Mask R-CNN
        print("[INFO] Mask R-CNN took {:.6f} seconds".format(end - start))
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
                # cv2.rectangle(clone, (startX, startY), (endX, endY), color, 2)

                # draw the predicted label and associated probability of the
                # instance segmentation on the image
                text = "{}: {:.4f}".format(LABELS[classID], confidence)
                cv2.putText(clone, text, (startX, startY - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                # show the output image

        self.output_array = clone.transpose(2, 0, 1)
        self.displayOutputImage()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    myWindow = MyWindow()
    myWindow.show()
    app.exec_() # 이벤트루프 생성