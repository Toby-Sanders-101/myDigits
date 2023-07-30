import os
import time
import myDigitsReaderHelp as hlp
import cv2
import pyautogui
import numpy as np

size = "40x27"
qd = hlp.getQD(size)

cam = cv2.VideoCapture(0)#dim=480x640

num = int(input("What number? "))
f = open("myDigits_"+size+".csv","a")
count = 0
while True:
	ptime = time.time()
	ret, image = cam.read()
	image = hlp.shrinkData(image,size)
	bigIm = 255*cv2.resize(image,(0,0),fx=qd,fy=qd)
	ctime = time.time()
	bigIm = cv2.putText(bigIm,"FPS: "+str(int(1/(ctime-ptime))),(75,50),cv2.FONT_HERSHEY_SIMPLEX,1,(255, 0, 0),2,cv2.LINE_AA)
	cv2.imshow('Imagetest',bigIm)
	k = cv2.waitKey(1)
	if k != -1:
#		print(len(image),'x',len(image[0]))
#		print(image)
		inp = pyautogui.confirm(text=str(count)+" already taken",title="Keep?",buttons=["Yes","No","Quit"])
		if inp=="Yes":
			string = ""
			count += 1
			for row in image:
				for col in row:
					string = string+str(col)+","
			string = string+str(num)+"\n"
			f.write(string)
		elif inp=="No":
			pass
		else:
			break
cam.release()
cv2.destroyAllWindows()
f.close()
