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

