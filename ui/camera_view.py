import cv2
from PyQt5.QtWidgets import QLabel
from PyQt5.QtGui import QImage, QPixmap
import time
from config.settings import CAMERA_SOURCE


class CameraView(QLabel):
    def __init__(self, width=640, height=480):
        super().__init__()
        self.setFixedSize(width, height)

        # Camera source comes from config
        self.cap = cv2.VideoCapture(CAMERA_SOURCE)

        if not self.cap.isOpened():
            raise RuntimeError(f"Unable to open camera source: {CAMERA_SOURCE}")

    def read_frame(self):
        if not self.cap.isOpened():
            print("[INFO] Camera disconnected, retrying...")
            time.sleep(1)
            self.cap.open(CAMERA_SOURCE)
            return None

        ret, frame = self.cap.read()
        if not ret:
            print("[WARN] Frame read failed, reconnecting...")
            self.cap.release()
            time.sleep(1)
            self.cap.open(CAMERA_SOURCE)
            return None

        return frame


    def display_frame(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qt_img = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        self.setPixmap(QPixmap.fromImage(qt_img))

    def switch_source(self, new_source):
        if self.cap.isOpened():
            self.cap.release()

        self.cap = cv2.VideoCapture(new_source)


    def release(self):
        if self.cap.isOpened():
            self.cap.release()
