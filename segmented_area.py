import cv2
import numpy as np
from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO('yolov8s-seg.pt')

# Predict the segmentation masks for the image
image = cv2.imread("../0.jpg")
masks = model(image)

# Get the segmentation mask for the desired object
desired_object_mask = masks[0][0]

# Extract the segmented area from the image using the segmentation mask
segmented_area = cv2.bitwise_and(image, image, mask=desired_object_mask)

# Display the segmented area
cv2.imshow("Segmented Area", segmented_area)
cv2.waitKey(0)
cv2.destroyAllWindows()
