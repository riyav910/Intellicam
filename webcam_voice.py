import cv2
import pyttsx3
import time
from ultralytics import YOLO
from collections import Counter
import os
from datetime import datetime
from collections import deque


# Load the model and TTS engine
model = YOLO('yolov8l.pt')
engine = pyttsx3.init()
engine.setProperty('rate', 150)

voice_enabled = True
detection_log = deque(maxlen=10)  # Holds last 10 detections

# Track announced objects and last detection time
announced_labels = set()
last_detection_time = time.time()

dangerous_objects = {"knife", "gun", "fire","hammer"}
screenshot_dir = "screenshots"
os.makedirs(screenshot_dir, exist_ok=True)

# Open webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, stream=True, verbose=False)
    # current_labels = set()
    label_list = []

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            label = model.names[cls]

            if conf > 0.5:
                color = (0, 255, 0)  # Default green

                if label in dangerous_objects:
                    color = (0, 0, 255)  # Red for danger
                    alert_text = f"⚠️ Alert: {label} detected"
                    print(alert_text)
                    if voice_enabled:
                        engine.say(alert_text)
                        engine.runAndWait()

                    # Save screenshot
                    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                    filename = f"{label}_{timestamp}.png"
                    filepath = os.path.join(screenshot_dir, filename)
                    cv2.imwrite(filepath, frame)
                    print(f"Screenshot saved: {filepath}")

                    with open("log.txt", "a") as log:
                        log.write(f"[{timestamp}] ALERT: {label} detected — Screenshot: {filename}\n")
                    
                # Draw bounding box and label
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                label_list.append(label)
                
                # Format log entry
                timestamp = datetime.now().strftime("%H:%M:%S")
                log_entry = f"[{timestamp}] {label}"

                if label in dangerous_objects:
                    log_entry += " ⚠️"

                detection_log.append(log_entry)

                # Optional: write to log.txt
                with open("log.txt", "a", encoding="utf-8") as log_file:
                    log_file.write(log_entry + "\n")

            
    # Count objects
    object_counts = Counter(label_list)
    # Clear terminal (for a live refresh feel)
    os.system('cls' if os.name == 'nt' else 'clear')

    # Print object count summary
    print(" | ".join([f"{label}: {count}" for label, count in object_counts.items()]))
    print(f"Voice: {'ON' if voice_enabled else 'OFF'}\n")

    # Print detection log
    print("Detection Log:")
    for entry in list(detection_log)[-5:]:
        print(entry)

    status_text = "Voice: ON" if voice_enabled else "Voice: OFF"
    y_offset = 20
    cv2.putText(frame, status_text, (10, y_offset + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 200), 2)

    y_offset += 25
    # Draw the counts on top-left corner of the frame
    for label, count in object_counts.items():
        text = f"{label}: {count}"
        cv2.putText(frame, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 255, 255), 2)
        y_offset += 25
    log_y = y_offset + 40
    cv2.putText(frame, "Detection Log:", (10, log_y), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (255, 255, 0), 2)
    log_y += 25

    for log_line in list(detection_log)[-5:]:  # Last 5 entries
        cv2.putText(frame, log_line, (10, log_y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (180, 180, 180), 1)
        log_y += 20

    current_labels = set(label_list)
    # Check and speak new labels
    new_labels = current_labels - announced_labels
    if new_labels:
        for label in new_labels:
            print(f"Speaking: {label}")
            if voice_enabled:
                engine.say(f"{label} detected")
                engine.runAndWait()
        announced_labels.update(new_labels)
        last_detection_time = time.time()  # Reset timer after new detection

    # If no new object for 30 seconds, clear the record
    if time.time() - last_detection_time > 15:
        announced_labels.clear()
        print("Cleared announced labels after 15 seconds of no new detections.")

    cv2.imshow("YOLOv8 Live Detection with Voice", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('v'):
        voice_enabled = not voice_enabled
        state = "enabled" if voice_enabled else "muted"
        print(f"🔊 Voice output {state}.")
        # Optional: give a voice confirmation too
        if voice_enabled:
            engine.say("Voice enabled")
        else:
            engine.say("Voice muted")
        engine.runAndWait()

    if key == ord('q'):
        break

    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break

cap.release()
cv2.destroyAllWindows()
