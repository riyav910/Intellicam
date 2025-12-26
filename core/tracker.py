import time
from collections import defaultdict, Counter

class ObjectTracker:
    def __init__(self, timeout):
        self.timeout = timeout
        self.last_seen = defaultdict(float)

    def update(self, detected_items):
        now = time.time()

        for item in detected_items:
            self.last_seen[item] = now

        # Remove stale objects
        self.last_seen = {
            k: v for k, v in self.last_seen.items()
            if now - v <= self.timeout
        }

        return Counter(detected_items)
