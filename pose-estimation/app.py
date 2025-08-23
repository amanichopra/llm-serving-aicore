from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse
import torch
import numpy as np
import cv2
import io

# Detect if GPU is available
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load YOLOv8 pose model and move to GPU
model = torch.jit.load("model/yolov8n-pose.torchscript.pt").to(device)
model.eval()

app = FastAPI(title="YOLOv8 Pose API with GPU")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Read image bytes
    image_bytes = await file.read()
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Convert to tensor and move to GPU
    img_tensor = torch.from_numpy(img).permute(2,0,1).unsqueeze(0).float().to(device)

    # Run pose estimation
    with torch.no_grad():
        results = model(img)  # YOLOv8 TorchScript model expects np.array input, handles internally

    # Draw keypoints on image
    for person_keypoints in results[0].keypoints.xy:
        for x, y in person_keypoints:
            cv2.circle(img, (int(x), int(y)), 5, (0, 255, 0), -1)

    # Encode image to PNG bytes
    _, buffer = cv2.imencode('.png', img)
    io_buf = io.BytesIO(buffer)

    return StreamingResponse(io_buf, media_type="image/png")
