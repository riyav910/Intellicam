# Intellicam AI 🔍🛡

A real-time AI-powered object detection and surveillance system using *YOLOv8, **OpenCV, and **pyttsx3*. It detects objects live from your webcam, announces them via voice, logs them, highlights dangerous ones, and even takes screenshots of critical alerts!

---

## 🚀 Features

-  Real-time object detection with YOLOv8
-  Voice announcements of new objects (toggle with v key)
-  Dangerous object alerts (e.g., knife, gun, fire, hammer)
-  Screenshot saving for threats
-  Object count overlay on webcam feed
-  Live detection log on screen and in log.txt
-  Clean terminal output

---

## Tech Stack

- *Python*
- *YOLOv8 (Ultralytics)*
- *OpenCV*
- *pyttsx3* (text-to-speech)
- *Counter, **deque, **datetime, **os* (standard Python libs)

---

## ⚙ Installation

> ⚠ Python 3.8+ recommended  
> Create and activate a virtual environment before starting (optional but recommended)

<pre> ```bash git clone https://github.com/riyav910/Intellicam.git cd visionguard-ai 
# Create virtual environment (optional) 
python -m venv venv 
venv\Scripts\activate # On Windows 
# source venv/bin/activate # On Linux/macOS 
# Install dependencies 
pip install -r requirements.txt 
# Run the app 
python webcam_voice.py ``` </pre>

## Controls:

- Press v to toggle voice output (on/off)
- Press q to quit the webcam stream

Screenshots are saved to the screenshots/ folder when dangerous objects are detected.
Logs are saved in log.txt with timestamps.

---

## Future Improvements

- Object tracking with DeepSORT
- Custom dataset training for workplace detection
- Telegram/Email alert system
- GUI or Streamlit dashboard
- Multi-camera support
- Cloud-based analytics dashboard

---

## Connect
Built with 💻 and 🤖 by Riya Verma  
If you're working on something similar or want to collaborate, feel free to connect!
