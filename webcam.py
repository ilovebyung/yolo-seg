import cv2
import os
import util

# Create a directory to save the images
if not os.path.exists('images'):
    os.makedirs('images')

# Initialize the camera
cap = cv2.VideoCapture(0)

# Loop until the user presses the Esc key
while True:
    # Capture a frame from the camera
    ret, frame = cap.read()

    # If the frame is empty, break out of the loop
    if not ret:
        break

    # Display the frame
    cv2.imshow('camera', frame)

    # Save the frame to a file
    filename = util.get_filename()
    path = os.path.join('images', f'{filename}')
    cv2.imwrite(path, frame)

    # Wait for a key press
    key = cv2.waitKey(1)

    # If the user presses the Esc key, break out of the loop
    if key == 27:
        break

# Release the camera
cap.release()

# Close all windows
cv2.destroyAllWindows()
