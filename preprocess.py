import torch
import cv2
import numpy as np
from torchvision import transforms

def preprocess(image):
    image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    original_height, original_width = image_cv.shape[:2]
    
    ratio = 640.0 / max(original_height, original_width)
    new_height = int(original_height * ratio)
    new_width = int(original_width * ratio)
    
    image_resized = cv2.resize(image_cv, (new_width, new_height))
    
    input_image = np.zeros((640, 640, 3), dtype=np.uint8)
    
    y_offset = (640 - new_height) // 2
    x_offset = (640 - new_width) // 2
    input_image[y_offset:y_offset + new_height, x_offset:x_offset + new_width] = image_resized
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    image_tensor = transform(input_image)
    image_tensor = image_tensor.unsqueeze(0)
    
    return image_tensor, image_cv, (original_width, original_height), (x_offset, y_offset, ratio)

def draw_predictions(image, predictions, classes, pad_info):
    height, width = image.shape[:2]
    x_offset, y_offset, ratio = pad_info
    
    for pred in predictions:
        x1, y1, x2, y2, conf, cls = pred
        if conf > 0.3:
     
            x1 = max(0, int((x1 - x_offset) / ratio))
            y1 = max(0, int((y1 - y_offset) / ratio))
            x2 = min(width, int((x2 - x_offset) / ratio))
            y2 = min(height, int((y2 - y_offset) / ratio))
            
            
            color = (0, 0, 255) 
            label = f"{classes[int(cls)]} {conf:.2f}"
            
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            
            (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(image, (x1, y1-label_height-10), (x1+label_width, y1), color, -1)
            cv2.putText(image, label, (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    return image 