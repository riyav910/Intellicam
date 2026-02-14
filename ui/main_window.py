import cv2
import os
import time
from datetime import datetime
from collections import Counter, defaultdict

from PyQt5.QtWidgets import (
    QWidget, QLabel, QTextEdit, QVBoxLayout,
    QHBoxLayout, QCheckBox, QPushButton
)
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer, Qt

from config.settings import (
    MODEL, DANGEROUS_OBJECTS,
    CONFIDENCE_THRESHOLD, DANGER_THRESHOLD,
    ENABLE_FEATURE_LOGGING, LABEL_VOCAB
)

from core.detector import ObjectDetector
from core.alerts import AlertManager
from core.danger_model import DangerModel
from core.scene_reasoner import SceneReasoner
from utils.data_logger import FeatureLogger
from modules.narrator import SceneNarrator


class IntellicamUI(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Intellicam - Object Detection")
        self.setGeometry(100, 100, 900, 700)

        self.init_ui()

        # -------- Camera (like original app.py) --------
        self.cap = cv2.VideoCapture(0)

        # -------- Core Systems --------
        self.detector = ObjectDetector(MODEL)
        self.alerts = AlertManager()
        self.danger_model = DangerModel(LABEL_VOCAB)
        self.feature_logger = FeatureLogger()

        self.narrator = SceneNarrator(
            cooldown=2,
            enable_voice=True
        )
        # self.narrator.speak_sentence("Voice system initialized.")

        self.scene_reasoner = SceneReasoner()

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

    # ==================================================
    # UI
    # ==================================================

    def init_ui(self):
        self.image_label = QLabel()
        self.image_label.setFixedSize(640, 480)

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)

        controls = QHBoxLayout()

        self.voice_checkbox = QCheckBox("Voice Alerts")
        self.voice_checkbox.setChecked(True)
        self.voice_checkbox.stateChanged.connect(self.toggle_voice)
        controls.addWidget(self.voice_checkbox)

        self.screenshot_checkbox = QCheckBox("Save Screenshots")
        self.screenshot_checkbox.setChecked(True)
        self.screenshot_checkbox.stateChanged.connect(self.toggle_screenshot)
        controls.addWidget(self.screenshot_checkbox)

        self.describe_button = QPushButton("Describe Scene (V)")
        self.describe_button.clicked.connect(self.describe_scene_now)
        controls.addWidget(self.describe_button)

        layout = QVBoxLayout()
        layout.addWidget(self.image_label)
        layout.addLayout(controls)
        layout.addWidget(QLabel("Detection Log"))
        layout.addWidget(self.log_text)

        self.setLayout(layout)

    # ==================================================
    # Keyboard (EXACTLY like working app.py)
    # ==================================================

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Q:
            self.close()

        if event.key() == Qt.Key_V:
            self.describe_scene_now()

    # ==================================================
    # Controls
    # ==================================================

    def toggle_voice(self, state):
        self.alerts.enable_voice = state == Qt.Checked
        self.narrator.enable_voice = state == Qt.Checked

    def toggle_screenshot(self, state):
        self.alerts.enable_screenshot = state == Qt.Checked

    # ==================================================
    # Scene Description
    # ==================================================

    def describe_scene_now(self):
        ret, frame = self.cap.read()
        if not ret:
            return

        def debug_callback(sentence):
            print("SCENE OUTPUT:", sentence)
            self.narrator.speak_sentence(sentence)

        self.scene_reasoner.describe_scene(
            frame.copy(),
            callback=debug_callback
        )


    # ==================================================
    # Frame Loop
    # ==================================================

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return

        detections = self.detector.detect(frame)
        detected_labels = []

        for det in detections:
            label = det["label"]
            conf = det["confidence"]

            if conf < CONFIDENCE_THRESHOLD:
                continue

            x1, y1, x2, y2 = det["bbox"]
            color = (0, 200, 0)

            bbox_area = (x2 - x1) * (y2 - y1)
            frame_area = frame.shape[0] * frame.shape[1]
            bbox_ratio = min(bbox_area / frame_area, 1.0)

            if conf > 0.6 and ENABLE_FEATURE_LOGGING:
                self.feature_logger.log(label.lower(), conf, bbox_ratio)

            danger_score = self.danger_model.predict(
                label.lower(), conf, bbox_ratio
            )

            is_danger = (
                label.lower() in DANGEROUS_OBJECTS
                or danger_score >= DANGER_THRESHOLD
            )

            if is_danger:
                color = (0, 0, 200)
                msg = self.alerts.handle_danger(label, conf, frame.copy())
                if msg:
                    self.log_text.append(msg)

            detected_labels.append(label.lower())

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                frame, f"{label} {conf:.2f}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
            )

        # -------- Show detected counts --------
        if detected_labels:
            counts = Counter(detected_labels)
            text = "Detected Objects:\n"
            for item, count in counts.items():
                bar = "█" * min(count, 20)
                text += f"{item:<10}: {bar} ({count})\n"
        else:
            text = "Detected Objects:\n- None"

        self.log_text.setPlainText(text)

        # -------- Display Frame --------
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        qt_img = QImage(rgb.data, rgb.shape[1], rgb.shape[0], QImage.Format_RGB888)
        self.image_label.setPixmap(QPixmap.fromImage(qt_img))

    def closeEvent(self, event):
        self.cap.release()
        cv2.destroyAllWindows()
