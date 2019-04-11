import cv2 as cv 
import numpy as np

def loadingBar(accuracy):
    load = accuracy * 2.70
    color = (0, 0, 255)
    image = np.zeros((50, 300, 3), np.uint8)
    cv.rectangle(image, (0, 0), (300, 50), (255, 255, 255), cv.FILLED)
    cv.rectangle(image, (15, 15), (int(load)+15, 35), color, cv.FILLED)
    return image

image = loadingBar(50)
cv.imshow("zeros", image)
cv.imwrite("cero.jpg", image)

