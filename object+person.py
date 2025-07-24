import cv2
import numpy as np
import random
import time
from ultralytics import YOLO

try:
    import pyrealsense2 as rs
    use_realsense = True
except ImportError:
    use_realsense = False # if using lapt cam then keep false if use external cam use true

# Load both models
custom_model = YOLO(r"C:\Users\Anushrii\Downloads\person detection-20250719T153046Z-1-001\person detection\runs\detect\train\weights\best.pt")
default_model = YOLO("yolov8s.pt")

custom_names = custom_model.names
default_names = default_model.names

# Random color per class
class_colors = {}
def get_color(cls):
    if cls not in class_colors:
        class_colors[cls] = tuple(random.randint(100, 255) for _ in range(3))
    return class_colors[cls]

# RealSense or webcam setup
if use_realsense:
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    pipeline.start(config)
else:
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise SystemExit("❌ Cannot access webcam")

# GUI
cv2.namedWindow("YOLOv8 Combo", cv2.WINDOW_NORMAL)
print("Press 'q' to quit")
time.sleep(1)

while True:
    # Read frame
    if use_realsense:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue
        frame = np.asanyarray(color_frame.get_data())
    else:
        ok, frame = cap.read()
        if not ok:
            break

    # Predict from both models
    results_custom = custom_model(frame, verbose=False)[0]
    results_default = default_model(frame, verbose=False)[0]

    all_results = [(results_custom, custom_names), (results_default, default_names)]

    for results, names in all_results:
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf)
            cls = int(box.cls)
            label = f"{names[cls]} {conf*100:.1f}%"
            color = get_color(names[cls])

            # Draw box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
            cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw, y1), color, -1)
            cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    cv2.imshow("YOLOv8 Combo", frame)

    if cv2.getWindowProperty("YOLOv8 Combo", cv2.WND_PROP_VISIBLE) < 1:
        break
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
if use_realsense:
    pipeline.stop()
else:
    cap.release()
cv2.destroyAllWindows()
