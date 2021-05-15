import tensorflow as tf
from tf_bodypix.api import download_model, load_model, BodyPixModelPaths
import cv2
from matplotlib import pyplot as plt
import numpy as np

bodypix_model = load_model(download_model(BodyPixModelPaths.MOBILENET_FLOAT_50_STRIDE_16))

img = cv2.imread('beach.jpg')
img = img[:480, :640, :]

# get vid cap device
cap = cv2.VideoCapture(0)

# loop through frame
while cap.isOpened():
    ret, frame = cap.read()

    # BodyPix Detections
    result = bodypix_model.predict_single(frame)
    mask = result.get_mask(threshold=0.5).numpy().astype(np.uint8)
    masked_image = cv2.bitwise_and(frame, frame, mask=mask)

    # Apply virtual background
    neg = np.add(mask, -1)
    inverse = np.where(neg==-1, 1, neg).astype(np.uint8)
    masked_background = cv2.bitwise_and(img, img, mask=inverse)
    final = cv2.add(masked_image, masked_background)

    # Show result to user on desktop
    cv2.imshow('BodyPix', final)

    # Break loop outcome
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release() # Releases webcam or capture device
cv2.destroyAllWindows() # Closes imshow frames
