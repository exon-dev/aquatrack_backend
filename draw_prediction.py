import cv2

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