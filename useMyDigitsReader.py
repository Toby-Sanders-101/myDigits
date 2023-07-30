import numpy as np
import cv2
import time
from csv import reader
import os
import myDigitsReaderHelp as hlp

size = "40x27"

parameters = hlp.readParams(size)
qd = hlp.getQD(size)

#x,y,_,_,_ = hlp.readDigits(size="120x80")

def linear_forward(A, W, b):
	Z = np.dot(W,A)+b
	assert(Z.shape == (W.shape[0], A.shape[1]))
	return Z

def sigmoid(Z):
	return 1/(1+np.exp(-Z))

def relu(Z):
	return Z*(Z>0)

def linear_activation_forward(A_prev,W,b,activation):
	Z = linear_forward(A_prev,W,b)
	if activation == "sigmoid":
		A = sigmoid(np.clip(Z,-25,25))
	elif activation == "relu":
		A = relu(Z)
	return A

def L_model_forward(X, parameters):
	A = X
	L = len(parameters) // 2
	for l in range(1, L):
		A_prev = A 
		A = linear_activation_forward(A_prev,parameters['W'+str(l)],parameters['b'+str(l)],"relu")
	AL = linear_activation_forward(A,parameters['W'+str(L)],parameters['b'+str(L)],"sigmoid")
	return AL

def predict(X,parameters):
	al=L_model_forward(X,parameters)
	preds = []
	confs = []
	for i in al.T:
		mark = 0
		maxi = 0
		for j in range(4):
			if maxi<i[j]:
				mark = j
				maxi = i[j]
		preds.append(mark)
		confs.append(np.max(i)/np.sum(i))
	return preds, confs

#print(predict(x,parameters)[0])

cam = cv2.VideoCapture(0)#dim=480x640
ptime = time.time()
while True:
	ret, image = cam.read()
	image = hlp.shrinkData(image,size)
	x = image.flatten().reshape(1,(round(480/qd)*round(320/qd)))
	image = 255*cv2.resize(image,(0,0),fx=qd,fy=qd)
	preds,confs = predict(x.T,parameters)
	image = cv2.putText(image,str(preds[0])+"    "+str(round(confs[0]*100,1))+"%",(75,50),cv2.FONT_HERSHEY_SIMPLEX,1,(0, 0, 0),2,cv2.LINE_AA)
	ctime = time.time()
	image = cv2.putText(image,"FPS: "+str(int(1/(ctime-ptime))),(75,100),cv2.FONT_HERSHEY_SIMPLEX,1,(0, 0, 0),2,cv2.LINE_AA)
	cv2.imshow('Imagetest',image)
	k = cv2.waitKey(10)
	ptime = time.time()
	if k != -1:
		break
cam.release()
cv2.destroyAllWindows()
