from PyQt5.QtGui import QColor, QImage, QPixmap
import numpy as np
import math

def get_fixed_pixmap(self, inputImageArray):
    """
    Resize displayed image.
    """
    # Array Type의 이미지를 Label에 출력 : Array -> QImage -> QPixmap -> 출력
    (RGB, input_image_h, input_image_w) = inputImageArray.shape            
    color = QColor() # QImage에 들어갈 값
    image_ratio_w = self.viewCol / input_image_w
    image_ratio_h = self.viewRow / input_image_h
    if image_ratio_w < image_ratio_h:
        image_ratio_h = image_ratio_w
    else:
        image_ratio_w = image_ratio_h
    new_c = int(input_image_w * image_ratio_w)
    new_r = int(input_image_h * image_ratio_h)
    viewImage = QImage(new_c, new_r, QImage.Format_RGB888)

    # 미리 계산된 self.outputArray를 PyQt용 객체인 QImage로 변환.
    for r in range(new_r): # r -> x , c -> y
        for c in range(new_c):
            # rgb -> pixel -> image -> pixmap
            r_ = math.floor(r/image_ratio_h)
            c_ = math.floor(c/image_ratio_w)
            try:
                color.setRgbF(inputImageArray[0][r_][c_]/255,
                              inputImageArray[1][r_][c_]/255,
                              inputImageArray[2][r_][c_]/255)
                viewImage.setPixel(c,r,color.rgb())
            except:
                continue
    # 2-2. QImage를 QPixmap으로 변환. 후 라벨에 출력
    # 참고: https://cnpnote.tistory.com/entry/PYTHON-numpy-%EB%B0%B0%EC%97%B4%EC%9D%84-PySide-QPixmap%EC%9C%BC%EB%A1%9C-%EB%B3%80%ED%99%98
    viewPixmap = QPixmap.fromImage(viewImage)
    return viewPixmap

# outputArray를 outputImage로 변환하고, 이를 또다시 outputPixmap으로 변환해서 출력 라벨에 보여준다
def display_output_image(self, nRow=0, nCol=0):
    """
    Display output image.
    """
    (RGB, nRow, nCol) = self.output_array.shape
    color = QColor()  # pixel
    self.output_image = QImage(nCol, nRow, QImage.Format_RGB888)

    for r in range(nRow):  # r -> x , c -> y
        for c in range(nCol):
            # rgb -> pixel -> image -> pixmap
            color.setRgbF(self.output_array[0][r][c] / 255,
                          self.output_array[1][r][c] / 255,
                          self.output_array[2][r][c] / 255)  # rgb -> pixel
            # color Pixel -> Image[x,y]
            self.output_image.setPixel(c, r, color.rgb())
    self.output_pixmap = QPixmap.fromImage(self.output_image)
    # 참고: https://cnpnote.tistory.com/entry/PYTHON-numpy-%EB%B0%B0%EC%97%B4%EC%9D%84-PySide-QPixmap%EC%9C%BC%EB%A1%9C-%EB%B3%80%ED%99%98
    # 3. 보이는 크기 조절
    if (self.output_image.width() > self.viewCol or 
        self.output_image.height() > self.viewRow):
        pm = get_fixed_pixmap(self, self.output_array)
        self.labelOutput.setPixmap(pm)
    else:
        self.labelOutput.setPixmap(self.output_pixmap)
    self.output_array = self.output_array.astype(np.int16)
    self.setStatusTip("출력 이미지 크기: " + str(nCol) + " x " + str(nRow))

def set_base_image(self):
    """
    Make output image to input image, for continue processing the image.
    """
    self.input_array = self.output_array.copy()
    self.input_image = self.output_image.copy()
    self.input_pixmap = self.output_pixmap.copy()
    pm = get_fixed_pixmap(self, self.input_array)
    self.labelInput.setPixmap(pm)