from PyQt5.QtWidgets import QInputDialog
from scipy import ndimage
import matplotlib
import numpy as np

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