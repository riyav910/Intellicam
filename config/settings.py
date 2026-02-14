from ultralytics import YOLO

MODEL_PATH = "yolov8n.pt"
MODEL = YOLO(MODEL_PATH)

# ================= Detection Thresholds =================

CONFIDENCE_THRESHOLD = 0.75
DANGER_THRESHOLD = 0.7

# ================= Alert System =================

ALERT_COOLDOWN = 5  # seconds

# ================= Logging =================

ENABLE_FEATURE_LOGGING = True

# ================= UI Refresh =================

UI_UPDATE_INTERVAL = 0.5  # seconds
DISPLAY_TIMEOUT= 3  # seconds   

# ================= Object Categories =================

DANGEROUS_OBJECTS = [
    "knife", "gun", "fire", "chainsaw", "smoke",
    "axe", "bomb", "sword", "grenade", "syringe"
]

LABEL_VOCAB = [
    "knife", "gun", "fire", "smoke",
    "person", "bottle", "phone", "book"
]


# ================= Camera Configuration =================

# Use 0 for laptop webcam
# Use IP stream URL for mobile camera
CAMERA_SOURCE = 0
# CAMERA_SOURCE = "http://172.18.16.224:8080/video"