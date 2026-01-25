import cv2
import os
import time
import pyttsx3
import threading

from config.settings import ALERT_COOLDOWN
from utils.time_utils import (
    current_time_str,
    current_datetime_str,
    current_datetime_filename
)


class AlertManager:
    def __init__(
        self,
        enable_voice=True,
        enable_screenshot=True,
        log_dir="logs",
        screenshot_dir="screenshots",
        cooldown_seconds=ALERT_COOLDOWN
    ):
        self.log_dir = os.path.abspath(log_dir)
        self.screenshot_dir = os.path.abspath(screenshot_dir)

        self.engine = pyttsx3.init()
        self.enable_voice = enable_voice
        self.enable_screenshot = enable_screenshot

        self.cooldown_seconds = cooldown_seconds

        # Track last alert time per object
        self.last_alert_time = {}

        # Directories
        self.log_dir = log_dir
        self.screenshot_dir = screenshot_dir

        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.screenshot_dir, exist_ok=True)

        self.log_file_path = os.path.join(
            self.log_dir, "dangerous_detections.txt"
        )

    # ------------------ Internal helpers ------------------

    def _is_new_event(self, label):
        """
        Returns True if this object has not triggered
        an alert within the cooldown window.
        """
        now = time.time()

        last_time = self.last_alert_time.get(label)

        if last_time is None:
            return True

        return (now - last_time) >= self.cooldown_seconds

    def _update_event_time(self, label):
        self.last_alert_time[label] = time.time()

    def _log_to_file(self, message):
        with open(self.log_file_path, "a", encoding="utf-8") as f:
            f.write(message + "\n")

    def _save_screenshot(self, label, frame):
        filename = f"{label}_{current_datetime_filename()}.png"
        path = os.path.join(self.screenshot_dir, filename)
        cv2.imwrite(path, frame)

    def _speak(self, label):
        self.engine.say(f"Dangerous item detected: {label}")
        self.engine.runAndWait()

    # ------------------ Public API ------------------

    def handle_danger(self, label, confidence, frame):
        label = label.lower()

        # Ignore duplicate events within cooldown
        if not self._is_new_event(label):
            return None

        # Mark this as a new event
        self._update_event_time(label)

        time_str = current_time_str()
        datetime_str = current_datetime_str()

        #  Voice alert
        if self.enable_voice:
            threading.Thread(
                target=self._speak,
                args=(label,),
                daemon=True
            ).start()

        # Screenshot (only once per event)
        if self.enable_screenshot:
            self._save_screenshot(label, frame)

        #  File log (only once per event)
        file_log = (
            f"[{datetime_str}] "
            f"DANGEROUS OBJECT: {label.upper()} | "
            f"Confidence: {confidence:.2f}"
        )
        self._log_to_file(file_log)

        # UI log message
        return f"[{time_str}] ⚠️ {label.upper()} detected ({confidence:.2f})"
