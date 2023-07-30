import numpy as np
import cv2
import time
from csv import reader
import os
import myDigitsReaderHelp as hlp

size = "40x27" #these are the dimensions of the images that you want to process
total_pixels = 40*27

parameters = hlp.readParams(size) #this reads the parameters stored in a file
qd = hlp.getQD(size) #the size used should correspond to a 'quality drop'. This is the decrease in resolution from your camera's quality to your size/quality eg
			#a drop from dimensions 480x320 -> 40x27 is division by 12. 480/12=40 and 320/12=27. Therefore qd = 12
outRange = 10 #outRange is the number of unique possible outputs. There are 10 digits so my range would be 10 however this may need altering if you want to change what content the program reads

def forward_one_layer(A_prev,W,b,activation): #calculates the outputs of a layer based on: its weights, biases, activation function and inputs (from the previous layer)
	Z = np.dot(W,A_prev)+b
	if activation == "sigmoid":
		A = 1/(1+np.exp(-(np.clip(Z,-25,25))))
	elif activation == "relu":
		A = max([0,Z])
	return A

def forward_all_layers(X, parameters): #moves through all the layers of the network and makes predictions using the inputs (X) and parameters
	A = X
	L = len(parameters) // 2
	for l in range(1, L):
		A_prev = A 
		A = forward_one_layer(A_prev,parameters['W'+str(l)],parameters['b'+str(l)],"relu")
	AL = forward_one_layer(A,parameters['W'+str(L)],parameters['b'+str(L)],"sigmoid")
	return AL

def predict(X,parameters): #makes a prediction for input data X. returns the most likely digit and its confidence
	al=forward_all_layers(X,parameters)
	preds = []
	confs = []
	for i in al.T:
		mark = 0 #index of highest confidence
		maximum = 0 #highest confidence
		for j in range(outRange):
			if maximum<i[j]:
				mark = j
				maximum = i[j]
		preds.append(mark)
		confs.append(np.max(i)/np.sum(i))
	return preds, confs

cam = cv2.VideoCapture(0) #initialises the camera
ptime = time.time()
while True:
	ret, image = cam.read() #take a photo
	image = hlp.shrinkData(image,size) #decrease the quality of the photo. I cropped it, shrunk it and reduced it to black and white (0 or 1)
	x = image.flatten().reshape(1,total_pixels) #reshape the image so that the model can process it
	image = 255*cv2.resize(image,(0,0),fx=qd,fy=qd) #this increases the size and changes it from 0's or 1's into 0's or 255's so that it is comprehendable
	preds,confs = predict(x.T,parameters) #use the model to make a prediction with a certain confidence
	image = cv2.putText(image,str(preds[0])+"    "+str(int(confs[0]*100))+"%",(75,50),cv2.FONT_HERSHEY_SIMPLEX,1,(0, 0, 0),2,cv2.LINE_AA) #output the prediction and confidence
	ctime = time.time()
	image = cv2.putText(image,"FPS: "+str(int(1/(ctime-ptime))),(75,100),cv2.FONT_HERSHEY_SIMPLEX,1,(0, 0, 0),2,cv2.LINE_AA) #output the fps
	cv2.imshow('Imagetest',image) #output the normal sized image
	k = cv2.waitKey(1) #doesn't return -1 if a key is pressed
	ptime = time.time()
	if k != -1: #exits program
		break
cam.release()
cv2.destroyAllWindows()
