from PyQt5.QtWidgets import QInputDialog
from scipy import ndimage
import matplotlib
import numpy as np
import math

def brightness_control(self):
    """
    Control image brightness.
    """
    # 얼마나 조절할지 입력
    intValue = QInputDialog.getInt(self,"test", "brightness(-255~255)")
    self.output_array = self.input_array + intValue[0]

    self.output_array = np.where(self.output_array > 255, 255,
        np.where(self.output_array < 0 , 0, self.output_array))

def binarize_image(self):
    """
    Binarize image.
    """
    # Threshold = 평균값
    Threshold = np.mean(self.input_array)
    self.output_array = self.input_array.copy()
    
    self.output_array[0] = (self.output_array[0] + self.output_array[1] + self.output_array[2]) / 3
    self.output_array[0] = np.where(self.output_array[0] > Threshold, 255, 0)
    self.output_array[1] = self.output_array[0].copy()
    self.output_array[2] = self.output_array[0].copy()

def masking_image(self):
    """
    Masking image using filter.
    """
    nRow = self.input_image.height()
    nCol = self.input_image.width()
    self.output_array = np.zeros((3,nRow,nCol))
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

    if maskSelector == 1: # 엠보싱
        outputHsvArray = matplotlib.colors.rgb_to_hsv(self.input_array.transpose(1, 2, 0))
        outputHsvArray[:,:,2] = ndimage.convolve(outputHsvArray[:,:,2], mask)
        self.output_array = matplotlib.colors.hsv_to_rgb(outputHsvArray).copy()
        self.output_array = self.output_array.transpose(2,0,1) + 127
    else:
        for RGB in range(3):
            self.output_array[RGB] = ndimage.convolve(self.input_array[RGB], mask)
    self.output_array = np.where(self.output_array > 255, 255, np.where(self.output_array < 0 , 0, self.output_array))

def resize_image(self):
    """
    Resize image.
    """
    inputText = QInputDialog.getText(self, "title", "Size(폭x넓이) ") 
    outputWidth, outputHeight = inputText[0].split("x")
    outputWidth = int(outputWidth)
    outputHeight = int(outputHeight)
    widthRatio = outputWidth / self.input_image.width()
    heightRatio = outputHeight / self.input_image.height()
    self.output_array = np.zeros((3,outputHeight, outputWidth))

    for RGB in range(3):
        for r in range(outputHeight):
            for c in range(outputWidth):
                r_ = math.floor(r/heightRatio)
                c_ = math.floor(c/widthRatio)
                value = self.input_array[RGB][r_][c_]
                self.output_array[RGB][r][c] = value

def rotate_image(self):
    """
    Rotate image using axis and degree.
    """     
    Rpos = QInputDialog.getText(self, "회전할 방향 입력", "방향: (x,y,z)")
    Rpos = Rpos[0]
    degree = QInputDialog.getText(self, "회전할 각도 입력", "각도: ")
    degree = float(degree[0])
    radian = degree * math.pi / 180
    nCol = self.input_image.width()
    nRow = self.input_image.height()
    self.output_array = np.zeros((3,nRow,nCol))
    cos = math.cos(radian)
    sin = math.sin(radian)        
    x_ = 0
    y_ = 0

    # 회전시 평행이동 이슈 -> 중점(x,y)을 회전시키고 그 점이 이동한만큼 다시 평행이동.
    originalCenter_x = nCol // 2
    originalCenter_y = nRow // 2
    (rotatedCenter_x, rotatedCenter_y, _) = \
        np.array([originalCenter_x, originalCenter_y, 1]) @ \
            np.array([[cos, -sin, 0], [sin, cos,  0], [0, 0, 1]])
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
                (x_, y_, _) = np.array([x, y, 1]) @ R                
                if 0 <= x_ < nCol and 0 <= y_ < nRow:
                    self.output_array[RGB, int(y_), int(x_)] = \
                        self.input_array[RGB, y, x]