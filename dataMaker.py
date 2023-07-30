import os
import time
import myDigitsReaderHelp as hlp
import cv2
import pyautogui
import numpy as np

size = "40x27" #these are the dimensions of the images that you want to process. Higher dimensions are better resolution but take longer to train and can result in too much 'noise'
qd = hlp.getQD(size) #the size used should correspond to a 'quality drop'. This is the decrease in resolution from your camera's quality to your size/quality eg
			#a drop from dimensions 480x320 -> 40x27 is division by 12. 480/12=40 and 320/12=27. Therefore qd = 12

cam = cv2.VideoCapture(0) #initialises the camera

num = int(input("What number? ")) #input the number that you are taking photos of
f = open("myDigits_"+size+".csv","a") #open the .csv file that you are using to store the data
count = 0 #this is the number of photos already taken
while True:
	ptime = time.time()
	ret, image = cam.read() #take a photo
	image = hlp.shrinkData(image,size) #decrease the quality of the photo. I cropped it, shrunk it and reduced it to black and white (0 or 1)
	bigIm = 255*cv2.resize(image,(0,0),fx=qd,fy=qd) #this increases the size and changes it from 0's or 1's into 0's or 255's so that it is comprehendable
	ctime = time.time()
	bigIm = cv2.putText(bigIm,"FPS: "+str(int(1/(ctime-ptime))),(75,50),cv2.FONT_HERSHEY_SIMPLEX,1,(255, 0, 0),2,cv2.LINE_AA) #output the current fps (frames per second)
	cv2.imshow('Imagetest',bigIm) #output the normal sized image
	k = cv2.waitKey(1) #returns -1 if a key is pressed
	if k != -1:
 		print(len(image),'x',len(image[0])) #output the dimensions of the image
 		print(image) #output the image itself
		inp = pyautogui.confirm(text=str(count)+" already taken",title="Keep?",buttons=["Yes","No","Quit"]) #creates a pop-upbox with three options
		if inp=="Yes": #this saves the image as a string in the file opened at the beginning
			string = ""
			count += 1
			for row in image:
				for col in row:
					string = string+str(col)+","
			string = string+str(num)+"\n"
			f.write(string)
		elif inp=="No": #this stops the image from being saved however continues running the program
			pass
		else: #this exits the program
			break
cam.release()
cv2.destroyAllWindows()
f.close()
