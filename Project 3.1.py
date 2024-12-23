import cv2
import numpy as np
from ultralytics import YOLO
import os

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
image = cv2.imread('C:\\Users\\super\\Documents\\GitHub\\Mo-s-Test\\Project 3 Data\\motherboard_image.JPEG',1)

gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

_, thresholded = cv2.threshold(gray_image, 110, 255, cv2.THRESH_BINARY)

inverted_mask=cv2.bitwise_not(thresholded)

contours, _ = cv2.findContours(inverted_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

largest_contour = max(contours, key=cv2.contourArea)
image_with_contours = image.copy()

mask = np.zeros_like(gray_image)
cv2.drawContours(mask, [largest_contour], -1, (255, 255, 255), thickness=cv2.FILLED)
cv2.drawContours(image_with_contours, [largest_contour], -1, (0, 255, 0), 2)

board_extracted = cv2.bitwise_and(image, image, mask=mask)

target_width, target_height = int(2172/2.5),int(2896/2.5)
resized_image = cv2.resize(image, (target_width, target_height))

cv2.imwrite('resized_image.JPEG',resized_image)
cv2.imwrite('inverted_mask.JPEG',inverted_mask)
cv2.imwrite('board_extracted.JPEG',board_extracted)
cv2.imwrite('image_with_contours.JPEG',image_with_contours)

# Step 2
'''
model = YOLO('yolov8n.pt')
model.train(data='C:\\Users\\super\\Documents\\GitHub\\Mo-s-Test\\Project 3 Data\\data\\data.yaml', epochs=150,
            imgsz=900, batch=2,device=0 ,name='Model_150Epoch')
'''


#Step 3
from PIL import Image
model =YOLO('C:\\Users\\super\\Documents\\GitHub\\Mo-s-Test\\Project 3 Data\\best.pt')

results1 = model('C:\\Users\\super\\Documents\\GitHub\\Mo-s-Test\\Project 3 Data\\data\\evaluation\\ardmega.jpg')
results2 = model('C:\\Users\\super\\Documents\\GitHub\\Mo-s-Test\\Project 3 Data\\data\\evaluation\\arduno.jpg')
results3 = model('C:\\Users\\super\\Documents\\GitHub\\Mo-s-Test\\Project 3 Data\\data\\evaluation\\rasppi.jpg')
# Evaluate the model

for r1 in results1:
    im_array = r1.plot()
    im = Image.fromarray(im_array[..., ::-1])
    im.show()
    im.save('ardmega_eval.jpg')

for r2 in results2:
    im_array = r2.plot()
    im = Image.fromarray(im_array[..., ::-1])
    im.show()
    im.save('arduno_eval.jpg')

for r3 in results3:
    im_array = r3.plot()
    im = Image.fromarray(im_array[..., ::-1])
    im.show()
    im.save('rasppi_eval.jpg')
