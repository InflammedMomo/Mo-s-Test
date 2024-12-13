import cv2
import numpy as np
from ultralytics import YOLO
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
image = cv2.imread('C:\\Users\\super\\Documents\\GitHub\\Mo-s-Test\\Project 3 Data\\motherboard_image.JPEG',1)

gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply thresholding to the image
_, thresholded = cv2.threshold(gray_image, 110, 255, cv2.THRESH_BINARY)

# Invert mask
inverted_mask=cv2.bitwise_not(thresholded)

# Perform edge detection using contour detection
contours, _ = cv2.findContours(inverted_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Find the largest contour (assuming it's the motherboard)
largest_contour = max(contours, key=cv2.contourArea)
image_with_contours = image.copy()

# Create an empty mask to extract the board
mask = np.zeros_like(gray_image)
cv2.drawContours(mask, [largest_contour], -1, (255, 255, 255), thickness=cv2.FILLED)
cv2.drawContours(image_with_contours, [largest_contour], -1, (0, 255, 0), 2)

# Extract the motherboard from the background
board_extracted = cv2.bitwise_and(image, image, mask=mask)

#Image resizing
target_width, target_height = int(2172/2.5),int(2896/2.5)
resized_image = cv2.resize(image, (target_width, target_height))

# Display the results
cv2.imwrite('resized_image.JPEG',resized_image)
cv2.imwrite('inverted_mask.JPEG',inverted_mask)
cv2.imwrite('board_extracted.JPEG',board_extracted)


# Step 2
import torch
'''
print(torch.cuda.is_available())
print(torch.cuda.device_count())
print(torch.cuda.current_device())
print(torch.cuda.device(0))
print(torch.cuda.get_device_name(0))'''

model = YOLO('yolov8n.pt')
model.train(data='C:\\Users\\super\\Documents\\GitHub\\Mo-s-Test\\Project 3 Data\\data\\data.yaml', epochs=150,
            imgsz=1000, batch=4,name='Model_150Epoch')

