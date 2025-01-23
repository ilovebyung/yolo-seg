import cv2
from ultralytics import YOLO
import numpy as np

# Load the YOLOv8n model
#model = YOLO("yolov8n.pt")
model = YOLO("yolo11n.pt")

# Export the model
# model.export(format="onnx")

# Initialize the webcam
cap = cv2.VideoCapture(0)
<<<<<<< HEAD
speeds = []
<<<<<<< HEAD
=======
>>>>>>> 177f21fd9ac76f58dd864e07510abd02eaf94b87
=======
>>>>>>> 1f51903 (speed)

while True:
    # Read a frame from the webcam
    ret, frame = cap.read()
    if not ret:
        break

<<<<<<< HEAD
    # Run YOLO inference on the frame
    results = model(frame, imgsz=320)
    speeds.append(results[0].speed['inference'])
=======
    # Run YOLOv8 inference on the frame
    results = model(frame, imgsz=320)
>>>>>>> 177f21fd9ac76f58dd864e07510abd02eaf94b87

    # Visualize the results on the frame
    for result in results:
        boxes = result.boxes.cpu().numpy()
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].astype(int)
            conf = box.conf[0]
            cls = int(box.cls[0])
            
            # Draw red bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            
            # Optionally, you can add label and confidence
            label = f'{model.names[cls]} {conf:.2f}'
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # Display the frame
<<<<<<< HEAD
    cv2.imshow("YOLO Inference", frame)
=======
    cv2.imshow("YOLO 11 Inference", frame)
>>>>>>> 1f51903 (speed)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()
<<<<<<< HEAD
<<<<<<< HEAD
print(sum(speeds)/len(speeds))
=======
>>>>>>> 177f21fd9ac76f58dd864e07510abd02eaf94b87
=======
print("YOLO 11 Inference", sum(speeds)/len(speeds))
>>>>>>> 1f51903 (speed)
