import time
import pyttsx3
import os
from datetime import datetime

class SceneNarrator:
    def __init__(
        self,
        cooldown=2,                 
        enable_voice=False,         
        log_file="logs/narration_log.txt"
    ):
        self.cooldown = cooldown
        self.enable_voice = enable_voice

        self.engine = pyttsx3.init()

        self.last_spoken_time = 0
        self.last_sentence = ""

        os.makedirs("logs", exist_ok=True)
        self.log_file = log_file

    # ---------------- private helpers ----------------

    def _timestamp(self):
        return datetime.now().strftime("%H:%M:%S")

    def _log_to_file(self, sentence):
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(f"[{self._timestamp()}] {sentence}\n")

    def _speak(self, sentence):
        self.engine.say(sentence)
        self.engine.runAndWait()


    # ---------------- public API ----------------

    def describe(self, counts_dict):
        now = time.time()

        # 🔹 1–2 second interval control
        if now - self.last_spoken_time < self.cooldown:
            return

        if not counts_dict:
            return

        parts = []
        for label, count in counts_dict.items():
            parts.append(f"{count} {label}")

        sentence = "I see " + ", ".join(parts)

        # 🔹 avoid repeating same sentence
        if sentence == self.last_sentence:
            return

        self.last_sentence = sentence
        self.last_spoken_time = now

        # 🔹 ALWAYS log
        self._log_to_file(sentence)

        # 🔹 optional speech
        if self.enable_voice:
            self._speak(sentence)
