import cv2
import numpy as np
from ultralytics import YOLO

class Detector:
    def __init__(self, model_path='yolov8n.pt'):
        self.model = YOLO(model_path)
        # self.cap = cv2.VideoCapture(0)
        self.cap = cv2.VideoCapture('park.mp4')

    def detect_persons(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            results = self.model(frame)

            for result in results:
                boxes = result.boxes.cpu().numpy()
                for box in boxes:
                    # Get the coordinates
                    x1, y1, x2, y2 = box.xyxy[0].astype(int)

                    # Get the class and confidence
                    class_id = box.cls[0].astype(int)
                    conf = box.conf[0]
                    
                    # Draw bounding box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    
                    # Put class name and confidence
                    label = f'{self.model.names[class_id]} {conf:.2f}'
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            cv2.imshow('Person Detection', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    def release(self):
        self.cap.release()
        cv2.destroyAllWindows()

# Usage
detector = Detector()
detector.detect_persons()
detector.release()