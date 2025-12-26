import csv
import os
from utils.time_utils import current_datetime_str


class FeatureLogger:
    def __init__(self, log_dir="logs", filename="feature_log.csv"):
        self.log_dir = log_dir
        self.filepath = os.path.join(log_dir, filename)

        os.makedirs(self.log_dir, exist_ok=True)

        # Create file with header if it doesn't exist
        if not os.path.exists(self.filepath):
            with open(self.filepath, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "timestamp",
                    "label",
                    "confidence",
                    "bbox_area_ratio"
                ])

    def log(self, label, confidence, bbox_area_ratio):
        with open(self.filepath, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                current_datetime_str(),
                label,
                round(confidence, 4),
                round(bbox_area_ratio, 6)
            ])
