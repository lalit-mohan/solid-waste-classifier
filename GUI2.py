import cv2
from Classify import main
import numpy as np
import pandas as pd
from scipy import misc, ndimage

font                   = cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (250,250)
fontScale              = 1
fontColor              = (255,0,0)
lineType               = 2

def recycle(img2):
    clf = main()
    dataList2 = []
    img = cv2.imread(img2)
#     print(img.shape)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_new = misc.imresize(img, (28, 28))
    hist = cv2.calcHist([img_new],[0],None,[256],[0,256])
#     print(hist.shape)
    hist = hist.reshape(1,hist.shape[0])
    arr = np.array(hist)
    k = clf.predict(arr)
    img = cv2.putText(img2, k[0], bottomLeftCornerOfText, font, fontScale, fontColor, lineType)
    return img

video_capture = cv2.VideoCapture(0)
while True:
    # Capturing video
    _, frame = video_capture.read()
    # Passing individual frames of video
    canvas = classify(frame)
    # Showing frame returned from classify function
    cv2.imshow('Video', canvas)
    # Press q to exit webcam
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
video_capture.release()
cv2.destroyAllWindows()
