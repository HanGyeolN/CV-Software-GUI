from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
import numpy as np
import sqlite3
from srcs import cv_image

DB_PATH = "./PyQtCVImageDB.sqlite3"

def query_sqlite(sql):
    """
    Execute query language.
    """
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    ret = 0

    cur.execute(sql)
    ret = cur.fetchone()
    cur.close()
    con.close()
    return ret

def open_image_sqlite(self):
    """
    Load image from sqlite database.
    1. get image size info, blob from db
    2. set input_array using blob
    3. set QImage array using input_array
    4. set Pixmap using input_image
    """
    img_id = QInputDialog.getText(self,"","label 입력:")
    sql = f"SELECT * FROM ImageBlob_TBL WHERE img_label='{img_id[0]}'"
    nCol, nRow, label, blob = query_sqlite(sql)
    temp = np.frombuffer(blob, np.uint8)
    self.input_array = temp.reshape((3,nRow,nCol)).copy()
    color = QColor()
    width = nCol
    height = nRow
    self.input_image = QImage(width, height, QImage.Format_RGB888)

    for r in range(nRow):
        for c in range(nCol):
            color.setRgbF(self.input_array[0][r][c] / 255, self.input_array[1][r][c] / 255,
                            self.input_array[2][r][c] / 255)  # rgb -> pixel
            self.input_image.setPixel(c, r, color.rgb()) # Image -> Pixmap
        # https://cnpnote.tistory.com/entry/PYTHON-numpy-%EB%B0%B0%EC%97%B4%EC%9D%84-PySide-QPixmap%EC%9C%BC%EB%A1%9C-%EB%B3%80%ED%99%98
    self.input_pixmap = QPixmap.fromImage(self.input_image)
    self.labelInput.setPixmap(self.input_pixmap)

def save_image_sqlite(self):
    """
    Save image to sqlite database.
    """
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    sql = "CREATE TABLE IF NOT EXISTS ImageBlob_TBL" \
            "(img_width INT, img_height INT, img_label CHAR(50), img_blob BLOB)"
    label = QInputDialog.getText(self, "DB에 저장하기", "Label 입력")
    if label[1] == 0:
        return
    label = label[0]
    blob = self.output_array.astype(np.uint8).tobytes()
    channel ,nRow, nCol = self.output_array.shape

    cur.execute(sql)
    sql = "INSERT INTO ImageBlob_TBL VALUES "\
            "(" + str(nCol) +","+ str(nRow) +",'"+ label +"', ?)"
    cur.execute(sql,(blob,))
    con.commit()
    cur.close()
    con.close()

def open_image_file(self):
    """
    Load image from image file path.
    """
    filename =  QFileDialog.getOpenFileName()
    # 2. 이미지파일 -> QPixmap
    self.input_pixmap = QPixmap(filename[0])
    nCol = width = self.input_pixmap.width()
    nRow = height = self.input_pixmap.height()
    # 3. QPixmap -> QImage : 픽셀값 받아오기
    self.input_image = self.input_pixmap.toImage()
    # 4. input Image에 numpy 배열로 넣기
    channel = 3
    self.input_array = np.zeros((channel,nRow,nCol)).copy()

    for r in range(nRow):
        for c in range(nCol):
            val = self.input_image.pixel(c,r)
            # pixmap -> image -> pixel -> rgb
            colors = QColor(val).getRgbF()            
            for RGB in range(channel):
                self.input_array[RGB,r,c] = int(colors[RGB]*255)
    if self.input_image.width() > self.viewCol or self.input_image.height() > self.viewRow:
        pm = cv_image.get_fixed_pixmap(self, self.input_array)
        self.labelInput.setPixmap(pm)
    else:
        self.labelInput.setPixmap(self.input_pixmap)
    self.setStatusTip("입력 이미지 크기: " + str(nCol) + " x " + str(nRow))

def save_image_file(self):
    """
    Save image to file path.
    """
    filename = QFileDialog.getSaveFileName()
    self.output_pixmap.save(filename[0])