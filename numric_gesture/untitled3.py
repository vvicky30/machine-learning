import numpy as np
import cv2
import argparse
from collections import deque


cap=cv2.VideoCapture(0)

pts = deque(maxlen=64)

Lower_green = np.array([110,50,50])
Upper_green = np.array([130,255,255])
while True:
	ret, img=cap.read()
	hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
	kernel=np.ones((5,5),np.uint8)
	mask=cv2.inRange(hsv,Lower_green,Upper_green)
	mask = cv2.erode(mask, kernel, iterations=2)
	mask=cv2.morphologyEx(mask,cv2.MORPH_OPEN,kernel)
	#mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,kernel)
	mask = cv2.dilate(mask, kernel, iterations=1)
	res=cv2.bitwise_and(img,img,mask=mask)
	cnts,heir=cv2.findContours(mask.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[-2:]
	center = None