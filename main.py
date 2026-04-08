from ultralytics import YOLO
import cv2
from deep_sort_realtime.deepsort_tracker import DeepSort

model = YOLO("yolov8n.pt")
tracker = DeepSort(max_age=30)

cap = cv2.VideoCapture("video.mp4")

width = int(cap.get(3))
height = int(cap.get(4))
fps = int(cap.get(5))

out = cv2.VideoWriter("output.mp4",
                      cv2.VideoWriter_fourcc(*'mp4v'),
                      fps,
                      (width, height))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)[0]

    detections = []

    for r in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = r
        if int(class_id) == 0:
            w = x2 - x1
            h = y2 - y1
            detections.append(([x1, y1, w, h], score, 'person'))

    tracks = tracker.update_tracks(detections, frame=frame)

    for track in tracks:
        if not track.is_confirmed():
            continue

        track_id = track.track_id
        l, t, w, h = map(int, track.to_ltrb())

        cv2.rectangle(frame, (l, t), (l + w, t + h), (0, 255, 0), 2)

        cv2.putText(frame, f"ID: {track_id}", (l, t - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow("Tracking", frame)

    out.write(frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
out.release()
cv2.destroyAllWindows()