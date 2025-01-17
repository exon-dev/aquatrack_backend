from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from preprocess import preprocess, draw_predictions
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
        yolov5_path = os.path.abspath('yolov5')
        if yolov5_path not in sys.path:
            sys.path.append(yolov5_path)
            
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
            
            if isinstance(results, (list, tuple)):
                pred = results[0]
            else:
                pred = results
                
            if len(pred.shape) == 3:
                pred = pred.squeeze(0)
                
            if pred.shape[1] != 6:
                boxes = pred[:, :4]  
                conf = pred[:, 4:5]  
                cls = pred[:, 5:]    
                cls_conf, cls_pred = torch.max(cls, 1, keepdim=True)
                pred = torch.cat((boxes, conf * cls_conf, cls_pred.float()), 1)
            
            conf_threshold = 0.3
            pred_boxes = pred.cpu().numpy()
            
            # Apply NMS with lower IOU threshold to keep more boxes
            indices = cv2.dnn.NMSBoxes(
                pred_boxes[:, :4].tolist(),
                pred_boxes[:, 4].tolist(),
                conf_threshold,
                nms_threshold=0.3  # Lower NMS threshold to keep more overlapping boxes
            )
            
            if len(indices) > 0:
                if isinstance(indices, tuple):
                    indices = indices[0]
                filtered_boxes = pred_boxes[indices]
                pred_boxes = filtered_boxes
            
        output_image = draw_predictions(original_image.copy(), pred_boxes, CLASSES, pad_info)
        
        output_dir = os.path.abspath('predictions')
        os.makedirs(output_dir, exist_ok=True)
        
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