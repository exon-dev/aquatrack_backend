from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from preprocess import preprocess
from draw_prediction import draw_predictions

import base64
from PIL import Image
import torch
import cv2
import numpy as np
import os
import sys
from collections import Counter
from pathlib import Path
import pathlib
from contextlib import contextmanager
import time
import argparse

yolov5_path = os.path.abspath('yolov5')
if yolov5_path not in sys.path:
    sys.path.append(yolov5_path)

from yolov5.utils.general import non_max_suppression, scale_boxes

# TODO: Change this to your own IP address 
MY_IP_ADDRESS = '192.168.1.2'


app = FastAPI()

origins = ['*']

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@contextmanager
def windows_compatible_path():
    """Context manager to temporarily redirect PosixPath to WindowsPath"""
    posix_backup = pathlib.PosixPath
    try:
        pathlib.PosixPath = pathlib.WindowsPath
        yield
    finally:
        pathlib.PosixPath = posix_backup

def load_model():
    try:
        model_path = os.path.abspath('models/exp10_best.pt')
        
        from models.common import DetectMultiBackend
        
        with windows_compatible_path():
            model = DetectMultiBackend(model_path, device='cpu')
            
        model.eval()
        return model
            
    except Exception as e:
        raise RuntimeError(f"Failed to load model: {str(e)}")

MODEL = load_model()
CLASSES = ['FIXED GALLONS']

@app.post('/api/v1/detect')
async def detect(file: UploadFile = File(...)):
    try:
        image = Image.open(file.file)
        image = image.convert('RGB')
        
        tensor_image, original_image, original_size, pad_info = preprocess(image)
        
        with torch.no_grad():
            results = MODEL(tensor_image)
            
            conf_thres = 0.2   
            iou_thres = 0.4     
            max_det = 12       
            
            pred = non_max_suppression(
                results, 
                conf_thres,
                iou_thres,
                classes=None,
                agnostic=False,
                max_det=max_det
            )[0]
            
            pred[:, :4] = scale_boxes(tensor_image.shape[2:], pred[:, :4], original_image.shape).round()
            
            final_boxes = []
            if len(pred) > 0:
                boxes = pred.cpu().numpy()
                
                boxes = boxes[boxes[:, 1].argsort()]
                
                y_threshold = 50 
                current_group = []
                grouped_boxes = []
                
                for box in boxes:
                    if not current_group or abs(box[1] - current_group[0][1]) < y_threshold:
                        current_group.append(box)
                    else:
                        current_group = sorted(current_group, key=lambda x: x[0])
                        grouped_boxes.extend(current_group)
                        current_group = [box]
                
                if current_group:
                    current_group = sorted(current_group, key=lambda x: x[0])
                    grouped_boxes.extend(current_group)
                
                for box in grouped_boxes:
                    if not final_boxes:
                        final_boxes.append(box)
                    else:
                        overlap = False
                        for existing_box in final_boxes:
                            iou = calculate_iou(box[:4], existing_box[:4])
                            if iou > 0.3:
                                overlap = True
                                break
                        if not overlap:
                            final_boxes.append(box)
            
            pred_boxes = np.array(final_boxes)
        
        output_image = draw_predictions(original_image.copy(), pred_boxes, CLASSES, pad_info)
        
        output_dir = os.path.abspath('predictions')
        os.makedirs(output_dir, exist_ok=True)
        
        _, buffer = cv2.imencode('.png', output_image)
        base64_image = base64.b64encode(buffer).decode('utf-8')
        base64_image_with_prefix = f"data:image/png;base64,{base64_image}"
        
        timestamp = int(time.time())
        output_filename = f'prediction_{timestamp}.jpg'
        output_path = os.path.join(output_dir, output_filename)
        
        cv2.imwrite(output_path, output_image)
        
        class_counts = Counter()
        for pred in pred_boxes:
            if pred[4] > 0.5:  
                class_idx = int(pred[5])
                class_counts[CLASSES[class_idx]] += 1
        
        formatted_counts = {
            class_name: count 
            for class_name, count in class_counts.items()
        }
        
        return {
            "status": 200,
            "message": "Prediction successful",
            "predictions": formatted_counts,
            "prediction_image": base64_image_with_prefix,
            "image_path": output_path,
            "image_shape": {
                "height": original_image.shape[0],
                "width": original_image.shape[1]
            },
            "detailed_predictions": [
                {
                    "class": CLASSES[int(pred[5])],
                    "confidence": float(pred[4]),
                    "bbox": pred[:4].tolist()
                }
                for pred in pred_boxes if pred[4] > 0.5
            ]
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get('/api/v1/image/{image_path}')
async def get_image(image_path: str):
    if os.path.exists(image_path):
        return FileResponse(image_path)
    raise HTTPException(status_code=404, detail="Image not found")

def calculate_iou(box1, box2):
    """Calculate IoU between two boxes"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    union = area1 + area2 - intersection
    return intersection / union if union > 0 else 0

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', default=MY_IP_ADDRESS, help='Host IP address')
    parser.add_argument('--port', default=5000, type=int, help='Port number')
    args = parser.parse_args()
    
    print(f"Running on {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)