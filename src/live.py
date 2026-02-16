from ultralytics import YOLO
import cv2
import os

# ================= CONFIG =================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "best.pt")
IMAGE_FOLDER = os.path.join(BASE_DIR, "data", "test")
CONF = 0.50
SCALE = 1.8

# ================= LOAD MODEL =================
model = YOLO(MODEL_PATH)

# ================= LOAD IMAGES =================
images = sorted([
    f for f in os.listdir(IMAGE_FOLDER)
    if f.lower().endswith((".jpg", ".png", ".jpeg"))
])

if not images:
    raise RuntimeError("No images found in folder")

index = 0
print("Controls: D = next | A = previous | Q = quit")

# ================= WINDOW SETUP =================
cv2.namedWindow("Solar Panel Prediction", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Solar Panel Prediction", 1200, 800)

# ================= MAIN LOOP =================
while True:
    img_path = os.path.join(IMAGE_FOLDER, images[index])
    img = cv2.imread(img_path)

    if img is None:
        index = (index + 1) % len(images)
        continue

    # -------- INFERENCE --------
    result = model(img, conf=CONF)[0]
    annotated = result.plot()

    # -------- RESIZE OUTPUT --------
    h, w = annotated.shape[:2]
    annotated = cv2.resize(
        annotated,
        (int(w * SCALE), int(h * SCALE)),
        interpolation=cv2.INTER_LINEAR
    )

    # -------- DISPLAY --------
    cv2.imshow("Solar Panel Prediction", annotated)

    key = cv2.waitKey(0) & 0xFF

    if key == ord('q'):
        break
    elif key == ord('d'):
        index = (index + 1) % len(images)
    elif key == ord('a'):
        index = (index - 1) % len(images)

cv2.destroyAllWindows()
