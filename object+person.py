import cv2
import numpy as np
import random
import time
from ultralytics import YOLO

# Try importing RealSense
try:
    import pyrealsense2 as rs
    use_realsense = True
except ImportError:
    use_realsense = False  # Set to False to open for laptop webcam

# Load all models
models = [
    YOLO(r"D:\Stationary (1)\runs (1)\detect (1)\train (1)\weights (1)\best (1).pt"),
    YOLO(r"D:\calendar\runs\detect\train4\weights\best.pt"),
    YOLO(r"D:\IDcard\runs\detect\train4\weights\best.pt"),
    YOLO(r"D:\pen\runs\detect\train3\weights\best.pt"),
    YOLO(r"D:\person detection-20250719T153046Z-1-001\person detection\runs\detect\train\weights\best.pt"),
    YOLO("yolov8s.pt")
]

# Color per class label
class_colors = {}
def get_color(cls_name):
    if cls_name not in class_colors:
        class_colors[cls_name] = tuple(random.randint(100, 255) for _ in range(3))
    return class_colors[cls_name]

# Setup RealSense or webcam
if use_realsense:
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    pipeline.start(config)
else:
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise SystemExit("❌ Cannot access webcam")

cv2.namedWindow("YOLOv8 Detection", cv2.WINDOW_NORMAL)
print("🔁 Press 'q' to quit...")

while True:
    # Get frame
    if use_realsense:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue
        frame = np.asanyarray(color_frame.get_data())
    else:
        ret, frame = cap.read()
        if not ret:
            break

    # Run all models
    for model in models:
        results = model(frame, verbose=False)[0]
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf)
            cls = int(box.cls)
            label_name = model.names[cls]
            label = f"{label_name} {conf*100:.1f}%"
            color = get_color(label_name)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
            cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw, y1), color, -1)
            cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    cv2.imshow("YOLOv8 Detection", frame)

    if cv2.getWindowProperty("YOLOv8 Detection", cv2.WND_PROP_VISIBLE) < 1:
        break
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
if use_realsense:
    pipeline.stop()
else:
    cap.release()
cv2.destroyAllWindows()

