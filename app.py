import sys
import cv2
import torch
import pyttsx3
import numpy as np
from datetime import datetime
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout, QTextEdit, QCheckBox
)
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer, Qt
from ultralytics import YOLO
from PyQt5.QtCore import QObject, pyqtSignal
from collections import defaultdict
import time
from collections import Counter


# Load YOLOv8 model
# model = YOLO("yolov8n.pt")
# model = YOLO("yolov8s.pt")
model = YOLO("yolov8m.pt")
# model = YOLO("yolov8ml.pt")

# Voice engine
engine = pyttsx3.init()

# Dangerous objects list
DANGEROUS_OBJECTS = ["knife", "gun", "fire", "chainsaw", "smoke", "axe", "bomb", "sword", "grenade", "syringe"]

# Voice alert toggle
ENABLE_ALERTS = True
ENABLE_SCREENSHOTS = True

class LogSignal(QObject):
    log_updated = pyqtSignal(str)


class IntellicamUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Intellicam - Object Detection")
        self.setGeometry(100, 100, 900, 700)

        self.init_ui()
        self.cap = cv2.VideoCapture(0)

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

        self.log_signal = LogSignal()
        self.log_signal.log_updated.connect(self.update_log)

        self.tracked_objects = defaultdict(lambda: 0)  # Object name → last seen timestamp
        self.display_timeout = 1.0  # seconds to wait before removing object if not seen

    def init_ui(self):
        # Camera label
        self.image_label = QLabel()
        self.image_label.setFixedSize(640, 480)

        # Detection log
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)

        # Toggle buttons
        self.voice_checkbox = QCheckBox("Voice Alerts")
        self.voice_checkbox.setChecked(True)
        self.voice_checkbox.stateChanged.connect(self.toggle_voice_alerts)

        self.screenshot_checkbox = QCheckBox("Save Screenshots")
        self.screenshot_checkbox.setChecked(True)
        self.screenshot_checkbox.stateChanged.connect(self.toggle_screenshots)

        # Layout
        top_layout = QVBoxLayout()
        top_layout.addWidget(self.image_label)

        control_layout = QHBoxLayout()
        control_layout.addWidget(self.voice_checkbox)
        control_layout.addWidget(self.screenshot_checkbox)

        bottom_layout = QVBoxLayout()
        bottom_layout.addLayout(control_layout)
        bottom_layout.addWidget(QLabel("Detection Log:"))
        bottom_layout.addWidget(self.log_text)

        main_layout = QVBoxLayout()
        main_layout.addLayout(top_layout)
        main_layout.addLayout(bottom_layout)

        self.setLayout(main_layout)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Q:
            self.close()

    def toggle_voice_alerts(self, state):
        global ENABLE_ALERTS
        ENABLE_ALERTS = state == Qt.Checked

    def toggle_screenshots(self, state):
        global ENABLE_SCREENSHOTS
        ENABLE_SCREENSHOTS = state == Qt.Checked

    def update_log(self, text):
        self.log_text.setPlainText(text)  # This replaces everything each time

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return

        results = model(frame)[0]
        boxes = results.boxes
        names = results.names
        detected_items = []

        for box in boxes:
            cls = int(box.cls[0])
            label = names[cls]
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            color = (0, 200, 0)

            if label.lower() in DANGEROUS_OBJECTS:
                color = (0, 0, 200)
                timestamp = datetime.now().strftime("%H:%M:%S")
                log_msg = f"[{timestamp}] ⚠️ {label.upper()} detected with {conf:.2f} confidence"
                # self.log_text.append(log_msg)
                self.log_text.append(log_msg)

                if ENABLE_ALERTS:
                    engine.say(f"Dangerous item detected: {label}")
                    engine.runAndWait()

                if ENABLE_SCREENSHOTS:
                    screenshot_filename = f"screenshot_{label}_{timestamp.replace(':', '-')}.png"
                    cv2.imwrite(screenshot_filename, frame)

            detected_items.append(label)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        current_time = time.time()

        # Update the last seen timestamp for currently detected items
        for item in detected_items:
            self.tracked_objects[item] = current_time

        # Remove items that haven't been seen recently
        to_remove = [item for item, last_seen in self.tracked_objects.items()
                    if current_time - last_seen > self.display_timeout]

        for item in to_remove:
            del self.tracked_objects[item]

        # Format display string
        # if self.tracked_objects:
        #     display_text = "Detected Objects:\n" + "\n".join(f"- {item}" for item in sorted(self.tracked_objects.keys()))
        # else:
        #     display_text = "Detected Objects:\n- None"
            
        # self.log_signal.log_updated.emit(display_text)

        if self.tracked_objects:
            # Count how many times each object appears in current frame
            counts = Counter(detected_items)
            
            display_text = "Detected Objects:\n"
            for item, count in counts.items():
                bar = "█" * min(count, 20)  # Limit bar length to 20
                display_text += f"{item:<10}: {bar} ({count})\n"
        else:
            display_text = "Detected Objects:\n- None"

        self.log_signal.log_updated.emit(display_text)

        # Convert to Qt image and display
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        qt_image = QImage(rgb_image.data, rgb_image.shape[1], rgb_image.shape[0], QImage.Format_RGB888)
        self.image_label.setPixmap(QPixmap.fromImage(qt_image))

    def closeEvent(self, event):
        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = IntellicamUI()
    window.show()
    sys.exit(app.exec_())
