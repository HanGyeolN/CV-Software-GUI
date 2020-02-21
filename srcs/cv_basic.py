from PyQt5.QtWidgets import QInputDialog
import numpy as np

def brightness_control(self):
    # 얼마나 조절할지 입력
    intValue = QInputDialog.getInt(self,"test", "brightness(-255~255)")
    self.output_array = self.input_array + intValue[0]

    self.output_array = np.where(self.output_array > 255, 255,
        np.where(self.output_array < 0 , 0, self.output_array))
