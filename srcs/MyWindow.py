from PyQt5.QtWidgets import QMainWindow
from PyQt5 import uic
from srcs import cv_files
from srcs import cv_image
from srcs import cv_basic
from srcs import cv_opencv

QT_UI_PATH = "./qt/cv_design.ui"
form_class = uic.loadUiType(QT_UI_PATH)[0]

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
        cv_opencv.equalize_histogram(self)
        self.displayOutputImage()

###### OpenCV
    def haarFaceDetection(self):
        cv_opencv.face_detection(self)
        self.displayOutputImage()

    def objectDetection(self):
        cv_opencv.object_detection(self)
        self.displayOutputImage()

    def maskRCNN(self):
        cv_opencv.object_detection_mask_rcnn(self)
        self.displayOutputImage()