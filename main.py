from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import base64
import torch
import cv2
import numpy as np
import os
import time
import argparse
from collections import Counter
import pathlib
from pathlib import Path
from preprocess import preprocess_image

pathlib.PosixPath = pathlib.WindowsPath

MY_IP_ADDRESS = '192.168.1.2'
PORT = 5000

app = FastAPI()

origins = ['*']
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def load_model():
    weights_path = os.path.join(os.getcwd(), 'models/final_training_best.pt')
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=weights_path, force_reload=True)
    model.eval()
    return model

MODEL = load_model()

"""
  These are the classes that the model is trained on.

  0: FIXED GALLON
  1: BROKEN GALLON
"""
CLASSES = ['FIXED GALLON', 'BROKEN GALLON']

@app.post('/api/v1/detect')
async def detect(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        np_img = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

        if image is None:
            raise ValueError("Invalid image format")

        image_rgb = preprocess_image(image)
        original_shape = image.shape

        results = MODEL(image_rgb)
        detections = results.xyxy[0].cpu().numpy()

        # adjust the treshold value
        # the lesser, more objects will be detected but its confidence value is lower
        # higher means less objects detected but higher confidence value (confident ang model)
        confidence_threshold = 0.1
        pred_boxes = [det for det in detections if det[4] > confidence_threshold]

        for pred in pred_boxes:
            x1, y1, x2, y2, conf, cls = pred
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            label = f"{CLASSES[int(cls)]} {conf:.2f}"
            color = (0, 255, 0) if int(cls) == 0 else (255, 0, 0)
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 4)
            cv2.putText(image, label, (x1, max(0, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        _, buffer = cv2.imencode('.png', image)
        base64_image = base64.b64encode(buffer).decode('utf-8')
        base64_image_with_prefix = f"data:image/png;base64,{base64_image}"

        output_dir = Path("predictions")
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = int(time.time())
        output_filename = f'prediction_{timestamp}.png'
        output_path = str(output_dir / output_filename)
        cv2.imwrite(output_path, image)

        class_counts = Counter(int(pred[5]) for pred in pred_boxes)
        formatted_counts = {CLASSES[idx]: count for idx, count in class_counts.items()}

        detailed_predictions = [
            {
                "class": CLASSES[int(pred[5])],
                "confidence": float(pred[4]),
                "bbox": [int(coord) for coord in pred[:4]]
            }
            for pred in pred_boxes
        ]

        print(formatted_counts)

        return {
            "status": 200,
            "message": "Prediction successful",
            "predictions": {
                "FIXED GALLON": formatted_counts['FIXED GALLON'] if 'FIXED GALLON' in formatted_counts else 0,
                "BROKEN GALLON": formatted_counts['BROKEN GALLON'] if 'BROKEN GALLON' in formatted_counts else 0
            },
            "prediction_image": base64_image_with_prefix,
            "image_path": output_path,
            "image_shape": {
                "height": original_shape[0],
                "width": original_shape[1]
            },
            "detailed_predictions": detailed_predictions
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get('/api/v1/image/{image_path}')
async def get_image(image_path: str):
    try:
        if os.path.exists(image_path):
            return FileResponse(image_path)
        else:
            raise HTTPException(status_code=404, detail="Image not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', default=MY_IP_ADDRESS, help='Host IP address')
    parser.add_argument('--port', default=PORT, type=int, help='Port number')
    args = parser.parse_args()
    
    print(f"Running on {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)
