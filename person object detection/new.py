import cv2
import time
from ultralytics import YOLO

# ─────────── USER SETTINGS ─────────── #
MODEL_PATH = r"C:\Users\Anushrii\Downloads\bestest.v2-roboflow-instant-2--eval-.yolov8-20250714T092940Z-1-001\bestest.v2-roboflow-instant-2--eval-.yolov8\runs\detect\train\weights\best.pt"
CONF_THRES = 0.50
IOU_THRES  = 0.35
MAX_DET    = 50
IMG_SIZE   = 960
# ───────────────────────────────────── #

model = YOLO(MODEL_PATH)
model.conf = CONF_THRES
names = model.model.names

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    raise SystemExit("❌ Cannot access webcam")

WIN = "YOLOv8 Detection"
cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
time.sleep(1)
print("✅ Press 'q' to quit")

while True:
    ok, frame = cap.read()
    if not ok:
        break

    results = model(
        frame,
        imgsz=IMG_SIZE,
        iou=IOU_THRES,
        max_det=MAX_DET,
        verbose=False
    )[0]

    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf)
        cls  = int(box.cls)
        label = f"{names[cls]} {conf * 100:.1f}%"

        # Color based on confidence
        if conf < 0.40:
            color = (0, 0, 255)      # Red
        elif conf < 0.70:
            color = (255, 255, 255)  # White
        else:
            color = (0, 255, 0)      # Green

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    cv2.imshow(WIN, frame)

    if cv2.getWindowProperty(WIN, cv2.WND_PROP_VISIBLE) < 1:
        break
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
