# Object Detection and Tracking System

This project detects and tracks multiple people in a video using YOLOv8 and DeepSORT.

## Features
- Detects people in video frames
- Assigns unique IDs to each person
- Tracks movement across frames
- Saves output video with bounding boxes and IDs

## Technologies Used
- Python
- YOLOv8 (Ultralytics)
- OpenCV
- DeepSORT

## How to Run
1. Install dependencies:
   pip install ultralytics opencv-python deep-sort-realtime

2. Place video file as video.mp4

3. Run:
   python main.py

## Output
- Output video saved as output.mp4