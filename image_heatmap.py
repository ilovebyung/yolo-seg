import cv2
from ultralytics import YOLO, solutions

# Load the YOLOv8 model
model = YOLO('yolov8n.pt')  # You can change this to other YOLOv8 models if needed

# Read the image
image = cv2.imread('bus.jpg')
h, w = image.shape[:2]  # image height and width

# Heatmap Init
heatmap_obj = solutions.Heatmap(
    colormap=cv2.COLORMAP_PARULA,
    view_img=True,
    shape="circle",
    names=model.names,
)

results = model.track(image, persist=True, show=True, classes=[0])
image = heatmap_obj.generate_heatmap(image, tracks=results)
cv2.imwrite("heatmap_output.png", image)

