from ultralytics import YOLO
from PIL import Image
import cv2
import matplotlib.pyplot as plt

model = YOLO('yolo11s.pt')
image = Image.open("park.jpg")
plt.imshow(image)
results = model.predict(source=image, show=True)  # save plotted images

for result in results:
    # Detection
    result.boxes.xyxy   # box with xyxy format, (N, 4)
    result.boxes.xywh   # box with xywh format, (N, 4)
    result.boxes.xyxyn  # box with xyxy format but normalized, (N, 4)
    result.boxes.xywhn  # box with xywh format but normalized, (N, 4)
    result.boxes.conf   # confidence score, (N, 1)
    result.boxes.cls    # cls, (N, 1)

    # # Segmentation
    # result.masks.data      # masks, (N, H, W)
    # result.masks.xy        # x,y segments (pixels), List[segment] * N
    # result.masks.xyn       # x,y segments (normalized), List[segment] * N

# Classification
prob = result.probs     # cls prob, (num_class, )

# Filter the results with a confidence score greater than 0.5
filtered_results = result.boxes[prob > 0.5]

# Each result is composed of torch.Tensor by default,
# in which you can easily use following functionality:
result = result.cuda()
result = result.cpu()
result = result.to("cpu")
result = result.numpy()

