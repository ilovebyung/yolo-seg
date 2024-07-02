from ultralytics import YOLO
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import numpy as np

model = YOLO('yolov8s-seg.pt')


'''
image 
'''
image = Image.open("dog.jpg")

# results = model.predict(source=image, save=True)  # save plotted images
results = model.predict(source=image)  

for result in results:
    # Segmentation
    data = result.masks.data      # masks, (N, H, W)
    mask = data[0]
    mask = mask.cpu()*255
    mask = mask.numpy()
    mask = mask.astype(np.uint8)
    # num_zeros = np.count_nonzero(mask == 0)
    detected_area = np.count_nonzero(mask == 255)
    # print(num_zeros)
    print(detected_area)


'''
webcam 
'''

# Initialize the camera
cap = cv2.VideoCapture(0)

# Loop until the user presses the Esc key
while True:
    # Capture a frame from the camera
    ret, frame = cap.read()
    results = model.predict(source=frame, classes=0) # person class

    # If the frame is empty, break out of the loop
    if not ret:
        break

    for result in results:
        print (result)

        # Segmentation
        data = result.masks.data      # masks, (N, H, W)
        mask = data[0]
        mask = mask.cpu()*255
        mask = mask.numpy()
        mask = mask.astype(np.uint8)
        # num_zeros = np.count_nonzero(mask == 0)
        detected_area = np.count_nonzero(mask == 255)
        # print(num_zeros)
        frame = cv2.putText(frame, f'detected_area: {detected_area}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX , 1, (255, 0, 0) , 2, cv2.LINE_AA) 

    # Display the frame
    cv2.imshow('detected_area', frame)

    # Wait for a key press
    key = cv2.waitKey(1)

    # If the user presses the Esc key, break out of the loop
    if key == 27:
        break

# Release the camera
cap.release()

# Close all windows
cv2.destroyAllWindows()
