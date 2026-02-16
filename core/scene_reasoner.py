import base64
import cv2
import tempfile
import os
import threading
import requests

OLLAMA_PATH = "C:\\Users\\riyav\\AppData\\Local\\Programs\\Ollama\\ollama.exe"
class SceneReasoner:
    """
    Uses Ollama + Moondream to generate
    natural scene descriptions from frames.
    """
    def __init__(self, model_name="moondream", ollama_path=OLLAMA_PATH):
        self.model_name = model_name
        self.ollama_path = ollama_path
        self.busy = False


    # ---------------- Internal Helper ----------------

    def _encode_image(self, frame):
        """
        Save frame temporarily and return path.
        """
        tmp_file = tempfile.NamedTemporaryFile(
            suffix=".jpg",
            delete=False
        )
        cv2.imwrite(tmp_file.name, frame, [cv2.IMWRITE_JPEG_QUALITY, 60])
        return tmp_file.name


    def _call_ollama(self, image_path):
        try:
            print("\n[SceneReasoner] Encoding image...")

            # Convert image to base64
            with open(image_path, "rb") as f:
                image_base64 = base64.b64encode(f.read()).decode("utf-8")

            payload = {
                "model": self.model_name,
                "prompt": "Describe image in 1 short sentence.",
                "images": [image_base64],
                "stream": False,
                "options": {
                    "num_ctx": 512,
                    "temperature": 0.2
                }
            }

            print("[SceneReasoner] Sending request to Ollama...")

            response = requests.post(
                "http://localhost:11434/api/generate",
                json=payload,
                timeout=60
            )

            print("[SceneReasoner] Status Code:", response.status_code)

            if response.status_code == 200:
                data = response.json()
                print("[SceneReasoner] RAW RESPONSE JSON:", data)

                sentence = data.get("response", "").strip()
                print("[SceneReasoner] FINAL OUTPUT:", sentence)

                return sentence
            else:
                print("[SceneReasoner] OLLAMA ERROR:", response.text)
                return ""

        except Exception as e:
            print("[SceneReasoner] Exception:", e)
            return ""


    # ---------------- Public API ----------------

    def describe_scene(self, frame, callback=None):
        """
        Runs description in background thread.
        """

        if self.busy:
            return

        self.busy = True

        def worker():
            # 🔹 Resize ONLY for LLM (faster processing)
            frame_small = cv2.resize(frame, (224, 224))

            image_path = self._encode_image(frame_small)
            description = self._call_ollama(image_path)

            if os.path.exists(image_path):
                os.remove(image_path)

            self.busy = False

            if callback:
                callback(description)

        threading.Thread(target=worker, daemon=True).start()


    