import torch
import cv2
import numpy as np
from torchvision import transforms

def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    shape = im.shape[:2]  
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  
        r = min(r, 1.0)

    ratio = r, r  
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1] 
    if auto: 
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)
    elif scaleFill:  
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0] 

    dw /= 2 
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, ratio[0], (dw, dh)

def preprocess(image):
    img = np.array(image)
    
    img = img[:, :, ::-1]
    
    original_image = img.copy()
    original_size = img.shape[:2]
    
    img, ratio, (dw, dh) = letterbox(img, new_shape=(640, 640), auto=True)
    
    img = img.transpose((2, 0, 1))  
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img)
    
    img = img.float()  
    
    if len(img.shape) == 3:
        img = img[None]
    
    pad_info = (dw, dh, ratio)
    
    return img, original_image, original_size, pad_info

