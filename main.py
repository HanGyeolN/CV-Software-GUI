import sys
from PyQt5.QtWidgets import QApplication
from srcs import MyWindow

if __name__ == "__main__":
    app = QApplication(sys.argv)
    myWindow = MyWindow.MyWindow()
    myWindow.show()
    app.exec_()