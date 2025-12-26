from ultralytics import YOLO

MODEL_PATH = "yolov8n.pt"
MODEL = YOLO(MODEL_PATH)

DANGEROUS_OBJECTS = [
    "knife", "gun", "fire", "chainsaw", "smoke",
    "axe", "bomb", "sword", "grenade", "syringe"
]

DISPLAY_TIMEOUT = 1.0

ENABLE_ALERTS = True
ENABLE_SCREENSHOTS = True

# ================= Camera Configuration =================

# Use 0 for laptop webcam
# Use IP stream URL for mobile camera
CAMERA_SOURCE = 0
# CAMERA_SOURCE = "http://172.18.16.224:8080/video"