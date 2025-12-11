# 🚦 Dynamic Signals: AI-Powered Smart Traffic Control

This project presents a smart traffic control system using AI and computer vision to dynamically manage traffic signals based on real-time vehicle detection from video inputs.

---

## 📌 Features

- 🧠 Vehicle detection using **YOLOv8 (Ultralytics)**
- 🚨 Emergency vehicle detection with prioritized signals
- 🎨 Interactive GUI built with **Tkinter**
- 📊 Real-time traffic signal control and status logs
- 📶 Dynamic green signal duration based on traffic density

---

## 🖼️ GUI Preview

<img src="signals_result.png" width="300">


---

## 📁 Project Structure


```
|-- main.py                 # Main GUI interface
|-- vehicle_detection.py    # Vehicle detection logic with YOLOv8
|-- centroid_tracker.py     # Centroid Tracker for tracking vehicles across frames
|-- signal_control.py       # Deprecated (legacy signal logic)
|-- signals_result.png      # Screenshot or sample traffic image
|-- Videos/                 # Traffic videos folder
|-- yolov3.cfg              # YOLO config
|-- yolov3.weights          # YOLO weights
|-- README.md               # This file
```

