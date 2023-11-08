from ultralytics import YOLO
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import numpy as np

file = 'sample.jpg'
# model = YOLO('yolov8s-seg.pt')
# image = Image.open("park.jpg")
model = YOLO('best.pt')
image = Image.open(file)
results = model.predict(source=image, conf=0.8)  # save plotted images

for result in results:
    # Segmentation
    data = result.masks.data      # masks, (N, H, W)
    xy = result.masks.xy        # x,y segments (pixels), List[segment] * N
    xyn = result.masks.xyn       # x,y segments (normalized), List[segment] * N

# get a sample of uint8

image = cv2.imread(file,0)
h, w = image.shape
plt.imshow(image, cmap='gray')
# cv2.imshow('image',image)
# cv2.waitKey(0)
# image = image.astype(np.float32)

# generate mask
mask = data[0]
# Convert the tensor to a NumPy array
mask = mask.cpu().numpy()*255

# resize image
mask = cv2.resize(mask, (w,h), interpolation = cv2.INTER_AREA)
mask = mask.astype(np.uint8)
plt.imshow(mask, cmap='gray')
# cv2.imshow('mask',mask)
# cv2.waitKey(0)

# Apply the mask to the image
masked_image = cv2.bitwise_and(image, mask)
plt.imshow(masked_image, cmap='gray')

cv2.imwrite('masked_image.jpg', masked_image)
masked_image = Image.open('masked_image.jpg')
masked_image.show()



