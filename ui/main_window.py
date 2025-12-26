import cv2
from PyQt5.QtWidgets import (
    QWidget, QLabel, QTextEdit, QVBoxLayout,
    QHBoxLayout, QCheckBox
)
from PyQt5.QtCore import QTimer, Qt

from ui.camera_view import CameraView
from config.settings import MODEL, DISPLAY_TIMEOUT, DANGEROUS_OBJECTS
from core.detector import ObjectDetector
from core.tracker import ObjectTracker
from core.alerts import AlertManager
from utils.logger import DetectionLogger
import time
from core.danger_model import DangerModel
from config.settings import CAMERA_SOURCE


CONFIDENCE_THRESHOLD = 0.75

class IntellicamUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Intellicam - Object Detection")
        self.setGeometry(100, 100, 900, 700)

        self.init_ui()

        self.detector = ObjectDetector(MODEL)
        self.tracker = ObjectTracker(DISPLAY_TIMEOUT)
        self.alerts = AlertManager()
        self.logger = DetectionLogger(self.log_text)

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

        self.ui_update_interval = 0.5  # seconds
        self.last_ui_update_time = 0

        self.label_vocab = [
            "knife", "gun", "fire", "smoke",
            "person", "bottle", "phone", "book"
        ]

        self.danger_model = DangerModel(self.label_vocab)

        self.DANGER_THRESHOLD = 0.7

        self.frame_count = 0
        self.fps = 0
        self.fps_timer_start = time.time()

    def init_ui(self):
        self.image_label = CameraView(640, 480)

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)

        # ---------- Create controls layout FIRST ----------
        controls = QHBoxLayout()

        self.voice_checkbox = QCheckBox("Voice Alerts")
        self.voice_checkbox.setChecked(True)
        self.voice_checkbox.stateChanged.connect(self.toggle_voice)
        controls.addWidget(self.voice_checkbox)

        self.screenshot_checkbox = QCheckBox("Save Screenshots")
        self.screenshot_checkbox.setChecked(True)
        self.screenshot_checkbox.stateChanged.connect(self.toggle_screenshot)
        controls.addWidget(self.screenshot_checkbox)

        # self.toggle_camera_checkbox = QCheckBox("Use Mobile Camera")
        # self.toggle_camera_checkbox.stateChanged.connect(self.toggle_camera)
        # controls.addWidget(self.toggle_camera_checkbox)

        # ---------- Main layout ----------
        layout = QVBoxLayout()
        layout.addWidget(self.image_label)
        layout.addLayout(controls)
        layout.addWidget(QLabel("Detection Log"))
        layout.addWidget(self.log_text)

        self.setLayout(layout)


    def toggle_voice(self, state):
        self.alerts.enable_voice = state == Qt.Checked

    def toggle_screenshot(self, state):
        self.alerts.enable_screenshot = state == Qt.Checked

    def toggle_camera(self, state):
        if state == Qt.Checked:
            mobile_url = CAMERA_SOURCE  # IP camera
            self.image_label.switch_source(mobile_url)
        else:
            self.image_label.switch_source(0)


    def update_frame(self):
        frame = self.image_label.read_frame()
        if frame is None:
            return

        detections = self.detector.detect(frame)
        labels = []

        for det in detections:
            label = det["label"]
            conf = det["confidence"]

            if conf < CONFIDENCE_THRESHOLD:
                continue

            x1, y1, x2, y2 = det["bbox"]
            color = (0, 200, 0)

            bbox_area = (x2 - x1) * (y2 - y1)
            frame_area = frame.shape[0] * frame.shape[1]
            bbox_area_ratio = bbox_area / frame_area
            
            danger_score = self.danger_model.predict(
                label.lower(),
                conf,
                bbox_area_ratio
            )

            if danger_score >= self.DANGER_THRESHOLD:
                color = (0, 0, 200)
                msg = self.alerts.handle_danger(label, conf, frame)
                if msg:
                    self.logger.log(msg)

            if danger_score >= self.DANGER_THRESHOLD:
                labels.append(label.lower())

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                frame, f"{label} {conf:.2f}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
            )

        counts = self.tracker.update(labels)
        current_time = time.time()
        if current_time - self.last_ui_update_time >= self.ui_update_interval:
            self.show_counts(counts)
            self.last_ui_update_time = current_time

        # FPS calculation
        self.frame_count += 1
        elapsed = time.time() - self.fps_timer_start

        if elapsed >= 1.0:
            self.fps = self.frame_count / elapsed
            self.frame_count = 0
            self.fps_timer_start = time.time()

        cv2.putText(
            frame,
            f"FPS: {int(self.fps)}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

        self.image_label.display_frame(frame)


    def show_counts(self, counts):
        if not counts:
            return

        text = "Detected Objects:\n"
        for item, count in counts.items():
            bar = "█" * min(count, 20)
            text += f"{item:<10}: {bar} ({count})\n"

        self.log_text.setPlainText(text)

    def closeEvent(self, event):
        self.image_label.release()
        cv2.destroyAllWindows()
