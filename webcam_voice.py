import cv2
import pyttsx3
import os
import time
from datetime import datetime
from ultralytics import YOLO
from collections import deque, Counter

from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QVBoxLayout, QHBoxLayout,
    QTextEdit, QCheckBox
)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import QTimer, Qt
import contextlib
import io

# Globals
model = YOLO("yolov8l.pt")
engine = pyttsx3.init()
engine.setProperty('rate', 150)

DANGEROUS_OBJECTS = {"knife", "gun", "fire", "hammer"}
voice_enabled = True
screenshot_enabled = True
announced_labels = set()
last_detection_time = time.time()
detection_log = deque(maxlen=10)
screenshot_dir = "screenshots"
os.makedirs(screenshot_dir, exist_ok=True)


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

    def init_ui(self):
        self.image_label = QLabel()
        self.image_label.setFixedSize(640, 480)

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)

        self.voice_checkbox = QCheckBox("Voice Alerts")
        self.voice_checkbox.setChecked(True)
        self.voice_checkbox.stateChanged.connect(self.toggle_voice_alerts)

        self.screenshot_checkbox = QCheckBox("Save Screenshots")
        self.screenshot_checkbox.setChecked(True)
        self.screenshot_checkbox.stateChanged.connect(self.toggle_screenshots)

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

    def toggle_voice_alerts(self, state):
        global voice_enabled
        voice_enabled = state == Qt.Checked

    def toggle_screenshots(self, state):
        global screenshot_enabled
        screenshot_enabled = state == Qt.Checked

    def update_frame(self):
        global announced_labels, last_detection_time
        ret, frame = self.cap.read()
        if not ret:
            return

        # results = model(frame, stream=True, verbose=False)
        with contextlib.redirect_stdout(io.StringIO()):
            results = model.predict(frame, stream = True, verbose = False, show=False)
        label_list = []

        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                label = model.names[cls]

                if conf > 0.5:
                    color = (0, 255, 0)
                    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

                    if label in DANGEROUS_OBJECTS:
                        color = (0, 0, 255)
                        alert_text = f"⚠️ Alert: {label} detected"
                        if voice_enabled:
                            engine.say(alert_text)
                            engine.runAndWait()

                        if screenshot_enabled:
                            filename = f"{label}_{timestamp}.png"
                            filepath = os.path.join(screenshot_dir, filename)
                            cv2.imwrite(filepath, frame)

                        with open("log.txt", "a") as log:
                            log.write(f"[{timestamp}] ALERT: {label} detected\n")

                    label_list.append(label)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                    # Update GUI log
                    log_entry = f"[{datetime.now().strftime('%H:%M:%S')}] {label}"
                    if label in DANGEROUS_OBJECTS:
                        log_entry += " ⚠️"
                    detection_log.append(log_entry)
                    # self.log_text.append(log_entry)
                    self.log_text.append(message)

        # Draw status and counts
        object_counts = Counter(label_list)
        y_offset = 20
        cv2.putText(frame, f"Voice: {'ON' if voice_enabled else 'OFF'}", (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 200), 2)
        y_offset += 25

        for label, count in object_counts.items():
            cv2.putText(frame, f"{label}: {count}", (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            y_offset += 25

        # Detection log preview on frame
        log_y = y_offset + 20
        cv2.putText(frame, "Detection Log:", (10, log_y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 255, 0), 2)
        log_y += 25
        for entry in list(detection_log)[-5:]:
            cv2.putText(frame, entry, (10, log_y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (180, 180, 180), 1)
            log_y += 20

        # Handle voice announcements
        current_labels = set(label_list)
        new_labels = current_labels - announced_labels
        if new_labels:
            for label in new_labels:
                if voice_enabled:
                    engine.say(f"{label} detected")
                    engine.runAndWait()
            announced_labels.update(new_labels)
            last_detection_time = time.time()

        if time.time() - last_detection_time > 15:
            announced_labels.clear()

        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        qt_image = QImage(rgb_image.data, rgb_image.shape[1], rgb_image.shape[0], QImage.Format_RGB888)
        self.image_label.setPixmap(QPixmap.fromImage(qt_image))

    def closeEvent(self, event):
        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    import sys
    app = QApplication(sys.argv)
    window = IntellicamUI()
    window.show()
    sys.exit(app.exec_())
