from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import base64
from ultralytics import YOLO
import numpy as np
import cv2
import io
import torch

# Detect device
device = 0 if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load YOLOv11 pose model
model = YOLO("./model/yolo11n-pose.pt")

# COCO skeleton (if model doesn’t provide one)
# Each tuple is (start_idx, end_idx) for keypoints
COCO_SKELETON = [
    (5, 7), (7, 9),       # left arm
    (6, 8), (8, 10),      # right arm
    (5, 6),               # shoulders
    (11, 13), (13, 15),   # left leg
    (12, 14), (14, 16),   # right leg
    (11, 12),             # hips
    (5, 11), (6, 12)      # torso
]

class ImagePayload(BaseModel):
    img_bytes: str  # This will contain the base64-encoded string

app = FastAPI(title="YOLOv11 Pose API with Ultralytics")

@app.post("/v2/predict")
async def predict_json(file: UploadFile = File(...)):
    # Read image bytes 
    image_bytes = await file.read()

    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Run inference
    results = model(img, device=device, verbose=False)

    for result in results:
        keypoints = result.keypoints.xy.cpu().numpy()  # (num_people, num_kpts, 2)

        for person in keypoints:
            # Draw keypoints
            for x, y in person:
                cv2.circle(img, (int(x), int(y)), 4, (0, 255, 0), -1)

            # Draw skeleton connections
            for i, j in COCO_SKELETON:
                if i < len(person) and j < len(person):
                    x1, y1 = person[i]
                    x2, y2 = person[j]
                    if x1 > 0 and y1 > 0 and x2 > 0 and y2 > 0:  # valid points
                        cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)

    # Encode back to PNG
    _, buffer = cv2.imencode('.png', img)
    io_buf = io.BytesIO(buffer)

    return StreamingResponse(io_buf, media_type="image/png")

@app.post("/v2/predictbs")
async def predict_json(payload: ImagePayload):
    img_bytes = base64.b64decode(payload.img_bytes)

    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Run inference
    results = model(img, device=device, verbose=False)

    for result in results:
        keypoints = result.keypoints.xy.cpu().numpy()  # (num_people, num_kpts, 2)

        for person in keypoints:
            # Draw keypoints
            for x, y in person:
                cv2.circle(img, (int(x), int(y)), 4, (0, 255, 0), -1)

            # Draw skeleton connections
            for i, j in COCO_SKELETON:
                if i < len(person) and j < len(person):
                    x1, y1 = person[i]
                    x2, y2 = person[j]
                    if x1 > 0 and y1 > 0 and x2 > 0 and y2 > 0:  # valid points
                        cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)

    # Encode back to PNG
    _, buffer = cv2.imencode('.png', img)
    io_buf = io.BytesIO(buffer)

    return StreamingResponse(io_buf, media_type="image/png")
