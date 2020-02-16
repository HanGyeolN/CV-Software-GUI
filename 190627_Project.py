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


# pip install pillow numpy PyQt5 
# Anaconda/Library/bin/qt_designer.exe
# ui파일 저장경로

UIPath = "C:/Users/ssna/Desktop/PyQt2/qt_design_test.ui"

form_class = uic.loadUiType(UIPath)[0]

class MyWindow(QMainWindow, form_class):
    HAAR_PATH = "C:/Users/ssna/Desktop/PyQt2/haar/"
    DB_PATH = "C:/Users/ssna/Desktop/PyQt2/PyQtCVImageDB.sqlite3"
    
    inputPixmap = None # QPixmap Object Image
    inputImage = None # QImage Object (0~1 사이 값)
    inputArray = None # npArray Image (0~255 사이 값)

    outputPixmap = None
    outputImage = None
    outputArray = None

    viewRow = 500
    viewCol = 500

    #def setupUi(self):
        #self.setWindowIcon(QIcon('cat04_64.png')) 

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
        print("\nloadFromDB")
        start = time.time()
        con = sqlite3.connect(self.DB_PATH)
        cur = con.cursor()

        getValue = QInputDialog.getText(self,"","label 입력:")
        print(getValue)
        getValue = getValue[0]

        sql = "SELECT * FROM ImageBlob_TBL WHERE img_label='"+getValue+"'"
        print(sql)
        cur.execute(sql)

        data = cur.fetchone()
        #print(data);print("load sqlite");print(len(data));print(type(data))

        nCol, nRow, label, blob = data

        temp = np.frombuffer(blob, np.uint8)
        #print(temp)

        self.inputArray = temp.reshape((3,nRow,nCol)).copy()

        cur.close()
        con.close()

    ####################
        color = QColor()  # pixel
        # 4. Array -> Image
        width = nCol
        height = nRow

        self.inputImage = QImage(width, height, QImage.Format_RGB888)

        for r in range(nRow):  # r -> x , c -> y
            for c in range(nCol):
                # rgb -> pixel -> image -> pixmap

                color.setRgbF(self.inputArray[0][r][c] / 255, self.inputArray[1][r][c] / 255,
                              self.inputArray[2][r][c] / 255)  # rgb -> pixel

                # color Pixel -> Image[x,y]
                self.inputImage.setPixel(c, r, color.rgb())

        # Image -> Pixmap
        # https://cnpnote.tistory.com/entry/PYTHON-numpy-%EB%B0%B0%EC%97%B4%EC%9D%84-PySide-QPixmap%EC%9C%BC%EB%A1%9C-%EB%B3%80%ED%99%98
        self.inputPixmap = QPixmap.fromImage(self.inputImage)

        self.labelInput.setPixmap(self.inputPixmap)
    ####################



    def saveToDB(self):
        con = sqlite3.connect(self.DB_PATH)
        cur = con.cursor()

        sql = "CREATE TABLE IF NOT EXISTS ImageBlob_TBL" \
              "(img_width INT, img_height INT, img_label CHAR(50), img_blob BLOB)"
        cur.execute(sql)

        label = QInputDialog.getText(self, "DB에 저장하기", "Label 입력")
        if label[1] == 0:
            print("라벨을 입력하세요")
            return

        label = label[0]
        #self.outputArray = self.outputArray.astype(np.uint8)
        blob = self.outputArray.astype(np.uint8).tobytes()
        channel ,nRow, nCol = self.outputArray.shape

        sql = "INSERT INTO ImageBlob_TBL VALUES "\
              "(" + str(nCol) +","+ str(nRow) +",'"+ label +"', ?)"

        cur.execute(sql,(blob,))

        con.commit()
        cur.close()
        con.close()
        print('save SQLite')

    def getAdjustedPixmap(self, inputImageArray):
        # Array Type의 이미지를 Label에 출력 : Array -> QImage -> QPixmap -> 출력
        print("\ngetAdjustedPixmap")
        start = time.time()
        
        # 1. 변수선언 및 메모리 할당
        (RGB, inputHeight, inputWidth) = inputImageArray.shape
                
        color = QColor() # QImage에 들어갈 값

        outputWidth = self.viewCol
        outputHeight = self.viewRow
        print(outputWidth, outputHeight)

        widthRatio = outputWidth/inputWidth
        heightRatio = outputHeight/inputHeight
        print(widthRatio, heightRatio)

        if widthRatio < heightRatio:
            heightRatio = widthRatio
        else:
            widthRatio = heightRatio

        new_c = int(inputWidth * widthRatio)
        new_r = int(inputHeight * heightRatio)
        print(new_c, new_r)
        print(widthRatio, heightRatio)

        viewImage = QImage(new_c, new_r, QImage.Format_RGB888)
        
        # 2. 알고리즘 - 
        # 2-1. 미리 계산된 self.outputArray를 PyQt용 객체인 QImage로 변환.

        for r in range(new_r): # r -> x , c -> y
            for c in range(new_c):
                # rgb -> pixel -> image -> pixmap

                r_ = math.floor(r/heightRatio)
                c_ = math.floor(c/widthRatio)

                try:
                    color.setRgbF(inputImageArray[0][r_][c_]/255, inputImageArray[1][r_][c_]/255, inputImageArray[2][r_][c_]/255) # rgb -> pixel
                    # color Pixel -> Image[x,y]
                    viewImage.setPixel(c,r,color.rgb())
                except:
                    continue

        # 2-2. QImage를 QPixmap으로 변환. 후 라벨에 출력
        # 참고: https://cnpnote.tistory.com/entry/PYTHON-numpy-%EB%B0%B0%EC%97%B4%EC%9D%84-PySide-QPixmap%EC%9C%BC%EB%A1%9C-%EB%B3%80%ED%99%98
        viewPixmap = QPixmap.fromImage(viewImage)

        print("adjust time:", time.time()-start)

        return viewPixmap

    def out2in(self):
        self.inputArray = self.outputArray.copy()
        self.inputImage = self.outputImage.copy()
        self.inputPixmap = self.outputPixmap.copy()
        mp = self.getAdjustedPixmap(self.inputArray)
        self.labelInput.setPixmap(mp)
        
    def openImage(self):
        print("\nopenImage")        

        # 1. 이미지파일 경로 받기
        filename =  QFileDialog.getOpenFileName()

        start = time.time() # 성능 측정
        print(filename)

        # 2. 이미지파일 -> QPixmap
        self.inputPixmap = QPixmap(filename[0])
        print(self.inputPixmap)

        nCol = width = self.inputPixmap.width()
        nRow = height = self.inputPixmap.height()
        print(width, height)

        # 3. label에 pixmap 출력
        # self.labelInput.setPixmap(self.inputPixmap)

        # 4. QPixmap -> QImage : 픽셀값 받아오기
        self.inputImage = self.inputPixmap.toImage()

        # 5. 실습처럼 input Image에 numpy 배열로 넣기
        channel = 3
        self.inputArray = np.zeros((channel,nRow,nCol)).copy()

        for r in range(nRow):
            for c in range(nCol):
                val = self.inputImage.pixel(c,r)
                # pixmap -> image -> pixel -> rgb
                colors = QColor(val).getRgbF()
                
                for RGB in range(channel):
                    self.inputArray[RGB,r,c] = int(colors[RGB]*255)

        # 조정된 픽스맵 출력
        if self.inputImage.width() > self.viewCol or self.inputImage.height() > self.viewRow:
            pm = self.getAdjustedPixmap(self.inputArray)
            self.labelInput.setPixmap(pm)
        else:
            self.labelInput.setPixmap(self.inputPixmap)
        
        print("time: ",time.time()-start)

        self.setStatusTip("입력 이미지 크기: " + str(nCol) + " x " + str(nRow))
                
                # print ("(%s,%s) = %s" % (r, c, self.inputImage[:,r,c]))
        #############################################################

    def saveImage(self): # 일반 PNG 파일로 저장 
        print("\nsaveImage")

        # 1. 변수선언 및 메모리 할당 - 저장경로 지정
        filename = QFileDialog.getSaveFileName()
        start = time.time()
        print(filename)        
        
        # 2. 알고리즘 - Pixmap 저장
        self.outputPixmap.save(filename[0])

        print("time:", time.time()-start)

    
    # outputArray를 outputImage로 변환하고, 이를 또다시 outputPixmap으로 변환해서 출력 라벨에 보여준다
    def displayOutputImage(self, nRow=0, nCol=0):
        print("\ndisplayOutputImage")
        start = time.time()
        print("0")

        (RGB, nRow, nCol) = self.outputArray.shape
        color = QColor()  # pixel


        self.outputImage = QImage(nCol, nRow, QImage.Format_RGB888)

        for r in range(nRow):  # r -> x , c -> y
            for c in range(nCol):
                # rgb -> pixel -> image -> pixmap

                color.setRgbF(self.outputArray[0][r][c] / 255, self.outputArray[1][r][c] / 255,
                              self.outputArray[2][r][c] / 255)  # rgb -> pixel

                # color Pixel -> Image[x,y]
                self.outputImage.setPixel(c, r, color.rgb())
        self.outputPixmap = QPixmap.fromImage(self.outputImage)

        # 참고: https://cnpnote.tistory.com/entry/PYTHON-numpy-%EB%B0%B0%EC%97%B4%EC%9D%84-PySide-QPixmap%EC%9C%BC%EB%A1%9C-%EB%B3%80%ED%99%98
        # 3. 보이는 크기 조절
        if self.outputImage.width() > self.viewCol or self.outputImage.height() > self.viewRow:
            pm = self.getAdjustedPixmap(self.outputArray)
            self.labelOutput.setPixmap(pm)
        else:
            self.labelOutput.setPixmap(self.outputPixmap)

        print("time:", time.time()-start)
        self.outputArray = self.outputArray.astype(np.int16)
        self.setStatusTip("출력 이미지 크기: " + str(nCol) + " x " + str(nRow))

####### 화소처리
    def brightnessControl(self):
        print("\n brightness control")
        # 얼마나 조절할지 입력
        intValue = QInputDialog.getInt(self,"test", "brightness(-255~255)")
        start = time.time()
        print("input mean:",np.mean(self.inputArray))
        print(intValue)

        # 1. 변수선언 및 메모리 할당 - outputArray 브로드캐스팅
        self.outputArray = self.inputArray + intValue[0]

        # 2. 알고리즘
        self.outputArray = np.where(self.outputArray > 255, 255, np.where(self.outputArray < 0 , 0, self.outputArray))


        print("outputArray shape", self.outputArray.shape)
        print("out mean:",np.mean(self.outputArray)) # 밝기 평균
        print("brightnessControl time:", time.time() - start)  # 성능측정
        
        self.displayOutputImage()



    def binary(self):
        Threshold = np.mean(self.inputArray)
        nRow = self.inputImage.height()
        nCol = self.inputImage.width()
        self.outputArray = self.inputArray.copy()
        
        self.outputArray[0] = (self.outputArray[0] + self.outputArray[1] + self.outputArray[2]) / 3
        self.outputArray[0] = np.where(self.outputArray[0] > Threshold, 255, 0)
        self.outputArray[1] = self.outputArray[0].copy()
        self.outputArray[2] = self.outputArray[0].copy()

        self.displayOutputImage(nRow,nCol)


####### 화소 영역 처리
    def masking(self):
        print("\nMasking")
        start = time.time()

    # 1. 변수선언 및 메모리 할당 - 변경할 크기
        nRow = self.inputImage.height()
        nCol = self.inputImage.width()
        
        maskSelector = QInputDialog.getText(self, "마스크 선택","1. 엠보싱 2. 블러링 3. 샤프닝 4. 가우시안 5. 커스텀")
        maskSelector = int(maskSelector[0])

        if maskSelector == 1: # 엠보싱
            mask = np.array([[-1, 0, 0], [ 0, 0, 0], [ 0, 0, 1]])
        elif maskSelector == 2: # LPF(블러링)
            mask = np.array([[1/9, 1/9, 1/9], [1/9, 1/9, 1/9], [1/9, 1/9, 1/9]])
        elif maskSelector == 3: # HPF(샤프닝)
            mask = np.array([[-1/9, -1/9, -1/9], [ -1/9, 8/9, -1/9], [ -1/9, -1/9, -1/9]])
        elif maskSelector == 4: # 가우시안 필터
            mask = np.array([[1/16, 1/8, 1/16], [1/8, 1/4, 1/8], [1/16, 1/8, 1/16]])
        elif maskSelector == 5: # Custom
            mask = QInputDialog.getText(self, "마스크 입력","3x3 : 9개 입력. \n,로 구분")
            mask = mask[0]
            mask = mask.split(',')
            for i in range(len(mask)):
                mask[i] = int(mask[i])
            mask = np.array(mask)
            mask = mask.reshape((3,3))

        self.outputArray = np.zeros((3,nRow,nCol))
    
    # 2. 알고리즘 - 마스킹
        if maskSelector == 1: # 엠보싱
            outputHsvArray = matplotlib.colors.rgb_to_hsv(self.inputArray.transpose(1, 2, 0))
            print(outputHsvArray.shape)
            print(type(outputHsvArray))
            outputHsvArray[:,:,2] = ndimage.convolve(outputHsvArray[:,:,2], mask)
            print(np.mean(outputHsvArray[:,:,0]), np.mean(outputHsvArray[:,:,1]), np.mean(outputHsvArray[:,:,2]))

            #outputHsvArray[:,:,2]

            self.outputArray = matplotlib.colors.hsv_to_rgb(outputHsvArray).copy()
            self.outputArray = self.outputArray.transpose(2,0,1) + 127

            print(np.mean(self.outputArray[0]), np.mean(self.outputArray[1]), np.mean(self.outputArray[2]))

        else:
            for RGB in range(3):
                self.outputArray[RGB] = ndimage.convolve(self.inputArray[RGB], mask)

        self.outputArray = np.where(self.outputArray > 255, 255, np.where(self.outputArray < 0 , 0, self.outputArray))
        #self.outputArray = self.outputArray.astype(np.uint8)


        print("time:", time.time() - start)
        
        self.displayOutputImage(nRow,nCol)

####### 기하처리
    def resizing(self):
        print("\nresizing")
        inputText = QInputDialog.getText(self, "title", "Size(폭x넓이) ")
        start = time.time()
        print(inputText)

        # 1. 변수선언 및 메모리 할당 - 변경할 크기
        inputWidth = self.inputImage.width()
        inputHeight = self.inputImage.height() 

        outputWidth, outputHeight = inputText[0].split("x")
        outputWidth = int(outputWidth)
        outputHeight = int(outputHeight)

        widthRatio = outputWidth/inputWidth
        heightRatio = outputHeight/inputHeight        
        self.outputArray = np.zeros((3,outputHeight, outputWidth))
        
        # 2. 알고리즘 - 이미지 크기 변경
        for RGB in range(3):
            for r in range(outputHeight):
                for c in range(outputWidth):
                    r_ = math.floor(r/heightRatio)
                    c_ = math.floor(c/widthRatio)
                    value = self.inputArray[RGB][r_][c_]

                    self.outputArray[RGB][r][c] = value
        
        #self.outputArray.astype(np.uint8)
        self.displayOutputImage(outputHeight, outputWidth)       

        print("time:", time.time() - start)
    
    def rotation(self):
        print("\nrotate")
        radList = []
        
        #for _ in range(3):
        Rpos = QInputDialog.getText(self, "회전할 방향 입력", "방향: (x,y,z)")
        Rpos = Rpos[0]
        degree = QInputDialog.getText(self, "회전할 각도 입력", "각도: ")
        degree = float(degree[0])
        radian = degree * math.pi / 180
        #radList.append(radian)
        start = time.time()

        #        
        nCol = self.inputImage.width()
        nRow = self.inputImage.height()
        self.outputArray = np.zeros((3,nRow,nCol))


        cos = math.cos(radian)
        sin = math.sin(radian)        

        x_ = 0
        y_ = 0
        temp = 0

        # 회전시 평행이동 이슈 -> 중점(x,y)을 회전시키고 그 점이 이동한만큼 다시 평행이동.
        originalCenter_x = nCol//2
        originalCenter_y = nRow//2

        (rotatedCenter_x, rotatedCenter_y, temp) = np.array([originalCenter_x, originalCenter_y, 1]) @ np.array([[cos, -sin, 0],
                       [sin, cos,  0], [0, 0, 1]])
        
        move_x = originalCenter_x - rotatedCenter_x
        move_y = originalCenter_y - rotatedCenter_y

        Rx = np.array([[1, 0, 0],
                       [0, cos, -sin],
                       [0, sin,  cos]])

        Ry = np.array([[cos, 0, sin],
                       [0  ,  1,  0],
                       [-sin, 0,cos]])

        Rz = np.array([[cos, -sin, 0],
                       [sin, cos,  0],
                       [move_x, move_y,  1]]) # 평행이동 + 회전 
                       # new_x = ( x * cos + y * sin ) + move_x
                       # new_y = ( x *-sin + y * cos ) + move_y
                       # 
        if Rpos == 'x':
            R = Rx
        elif Rpos == 'y':
            R = Ry
        elif Rpos == 'z':
            R = Rz
        else:
            return   
        
        for RGB in range(3):
            for x in range(nCol):
                for y in range(nRow):
                    (x_, y_, temp) = np.array([x, y, 1]) @ R
                    
                    if 0 <= x_ < nCol and 0 <= y_ < nRow:
                        self.outputArray[RGB, int(y_), int(x_)] = self.inputArray[RGB, y, x]
        
        self.displayOutputImage()

    
    # 1. 변수선언 및 메모리 할당
        

        print("time:", time.time() - start)
        


####### 통계처리
    def histogram(self):
        print("\nhistogram")
        start = time.time()
        print("check potin")

        if self.outputArray is None:
            return
        print("check potin")
    # 1. 변수선언 및 메모리 할당 - RGB 히스토그램 배열
        inputHist = np.zeros((3,256))
        outputHist = np.zeros((3,256))

        channel = 3 # 채널갯수 (R, G, B)
            # self.inputArray.shape = (3,1080,1920) 채널, 높이, 너비
        inputWidth =self.inputArray.shape[2]
        inputHeight = self.inputArray.shape[1]
        outputWidth = self.outputArray.shape[2] # width = nCol = c = [2] = x
        outputHeight = self.outputArray.shape[1] # height = nRow = r = [1] = y
        print("iw ih ow oh:", inputWidth, inputHeight, outputWidth, outputHeight)
        print("in out shape:", self.inputArray.shape, self.outputArray.shape)
        
    # 2. 알고리즘 - 히스토그램 계산 / 시각화   
    #    for RGB in range(channel):
    #        for r in range(inputHeight):
    #            for c in range(inputWidth):
    #                inputHist[RGB][int(self.inputArray[RGB][r][c])] += 1
    #    11초 -> 0.04초
        #self.inputArray = self.inputArray.astype(np.uint8)
        #self.outputArray = self.outputArray.astype(np.uint8)
        for RGB in range(3):
            inputHist[RGB] = np.histogram(self.inputArray[RGB], bins=256)[0]
            outputHist[RGB] = np.histogram(self.outputArray[RGB], bins=256)[0]

        print("time:", time.time() - start) 
        
        fig = plt.figure()
        r = fig.add_subplot(3,2,1)
        g = fig.add_subplot(3,2,3)
        b = fig.add_subplot(3,2,5)
        r.plot(inputHist[0], color='r')
        g.plot(inputHist[1], color='g')
        b.plot(inputHist[2], color='b')

        r2 = fig.add_subplot(3,2,2)
        g2 = fig.add_subplot(3,2,4)
        b2 = fig.add_subplot(3,2,6)
        r2.plot(outputHist[0], color='r')
        g2.plot(outputHist[1], color='g')
        b2.plot(outputHist[2], color='b')
        plt.show()

    def histogramEqualization(self):
        print("\nHistogram Equalization")
        start = time.time()

        channel, nRow, nCol = self.inputArray.shape
        print(channel, nRow, nCol)

        self.outputArray = np.zeros((channel, nRow, nCol))
        print("o")


        temp = self.inputArray.astype(np.uint8)
        for RGB in range(channel):
            self.outputArray[RGB] = cv2.equalizeHist(temp[RGB])
        print("o")


        print("o")

        self.displayOutputImage(nRow, nCol)
        print("o")

        print("Histogram Equalization:", time.time() - start)
###### ML
    def haarFaceDetection(self):
        print("test")
        start = time.time()
        face_cascade = cv2.CascadeClassifier(self.HAAR_PATH+"haarcascade_frontalface_alt.xml")

        cvPhoto2 = self.inputArray.transpose(1, 2, 0).copy()
        cvPhoto2 = cvPhoto2.astype(np.uint8)
        print(cvPhoto2.shape)
        print(type(cvPhoto2))

        grey = cv2.cvtColor(cvPhoto2, cv2.COLOR_RGB2GRAY)
        face_rects = face_cascade.detectMultiScale(grey, 1.1, 5)
        print(face_rects)

        for (x, y, w, h) in face_rects:
            cv2.rectangle(cvPhoto2, (x, y), (x + w, y + h), (0, 255, 0), 3)

        print(cvPhoto2.shape)
        self.outputArray = cvPhoto2.transpose(2, 0, 1)
        self.displayOutputImage()
        self.setStatusTip("사람수: " + str(len(face_rects)) + " 위치: " + str(face_rects))

###### OpenCV
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

        image = self.inputArray.transpose(1,2,0).copy()
        image = image.astype(np.uint8)

        (h, w) = image.shape[:2]

        photo = image.copy()
        blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 0.007843, (300, 300), 127.5)

        print("[INFO] computing object detections...")
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
        self.outputArray = photo.transpose(2,0,1)
        self.displayOutputImage()

    def maskRCNN(self):
        MASK_RCNN_PATH = "C:/Users/ssna/Desktop/PyQt2/mask-rcnn/mask-rcnn-coco"
        CONF_VALUE = 0.5
        THRESHOLD_VALUE = 0.3
        VISUALIZE = 0


        labelsPath = "C:/Users/ssna/Desktop/PyQt2/mask-rcnn/mask-rcnn-coco/object_detection_classes_coco.txt"
        print(labelsPath)
        print("0")
        LABELS = open(labelsPath).read().strip().split("\n")
        print("0")

        # load the set of colors that will be used when visualizing a given
        # instance segmentation
        colorsPath = MASK_RCNN_PATH+"/colors.txt"
        print("0")
        COLORS = open(colorsPath).read().strip().split("\n")
        COLORS = [np.array(c.split(",")).astype("int") for c in COLORS]
        COLORS = np.array(COLORS, dtype="uint8")
        print("0")

        # derive the paths to the Mask R-CNN weights and model configuration
        weightsPath = "C:/Users/ssna/Desktop/PyQt2/mask-rcnn/mask-rcnn-coco/frozen_inference_graph.pb"
        configPath = "C:/Users/ssna/Desktop/PyQt2/mask-rcnn/mask-rcnn-coco/mask_rcnn_inception_v2_coco_2018_01_28.pbtxt"

        # load our Mask R-CNN trained on the COCO dataset (90 classes)
        # from disk
        print("[INFO] loading Mask R-CNN from disk...")
        net = cv2.dnn.readNetFromTensorflow(weightsPath, configPath)

        # load our input image and grab its spatial dimensions
        print("1")
        image = self.inputArray.transpose(1,2,0).copy()
        image = image.astype(np.uint8)
        print("2")
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
                print("0")
                if VISUALIZE > 0:
                    print("1")
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

        self.outputArray = clone.transpose(2, 0, 1)
        self.displayOutputImage()




if __name__ == "__main__":
    app = QApplication(sys.argv)
    myWindow = MyWindow()
    myWindow.show()
    app.exec_() # 이벤트루프 생성