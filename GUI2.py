import cv2
from Classify import main
import numpy as np
import pandas as pd
from scipy import misc, ndimage
from tkinter import *
import tkinter.messagebox
root = Tk()      
canvas = Canvas(root, width = 700, height = 550,bg="black")      
canvas.pack()   
cam = cv2.VideoCapture(0)

cv2.namedWindow("test")

img_counter = 0
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
    tkinter.messagebox.showinfo( "Result",k)

while True:
    ret, frame = cam.read()
    cv2.imshow("test", frame)
    if not ret:
        break
    k = cv2.waitKey(1)

    if k%256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break
    elif k%256 == 32:
        # SPACE pressed
        img_name = "opencv_frame_{}.png".format(img_counter)
        cv2.imwrite(img_name, frame)
        print("{} written!".format(img_name))
        img_counter += 1
        img = PhotoImage(file=img_name)      
        canvas.create_image(30,30, anchor=NW, image=img)
        check=Button(root,text="TEST",command=recycle(img_name),background="black",fg="white")
        check.pack()
        root.mainloop()    

cam.release()

cv2.destroyAllWindows()
