import numpy as np
import cv2
import time
from csv import reader
import os
import sys
np.set_printoptions(linewidth=np.inf)

n_layers = 3

def getQD(size): #key for retrieving quality drop
	qdDict = {"40x27":12}
	qd = qdDict[size]
	return qd

def shrinkData(inp,size): #reduces data size and quality
	qd = getQD(size)
	cropped = inp[0:480,160:480]
	black_n_white = cv2.cvtColor(cropped,cv2.COLOR_BGR2GRAY)
	lossy_compressed = cv2.resize(black_n_white,(0,0),fx=1/qd,fy=1/qd)
	out = np.uint8(lossy_compressed>127) #reduces 0-255 -> 0's or 1's
	return out #output low data version
	
def readParams(size): #reads the parameters saved in a file
	layers = {}
	f = open("myParameters_"+size+".txt","r")
	for l in range(n_layers*2):
		arrarr = []
		string = f.readline().replace(",","").replace("[","").replace("]]","").replace("\n","")
		list1 = string.split("]")
		for item1 in list1:
			list2 = item1.split(" ")
			arr = []
			for item2 in list2:
				if item2!="":
					arr.append(float(item2))
			arrarr.append(np.array(arr))
		layers[["W","b"][l%2]+str(l//2+1)] = np.array(arrarr)
	f.close()
	parameters = layers
	print("Parameters have been loaded")
	return parameters

def readDigits(size,test=0,outRange=10): #creates testing and training data
	x=[]
	y=[]
	tx=[]
	ty=[]
	f = open("myDigits_"+size+".csv", 'r')
	csv_reader = reader(f)
	count = 0
	for row in csv_reader:
		if not row:
			continue
		try:
			number = count%(int(1/test)) #if test!=0, do the modulus operation
		except:
			number = 1 #anything except 0
		if number==0: #add this row to the testing data rather than the training data
			ty.append(int(row.pop()))
			introw = []
			for item in row:
				introw.append(float(item))
			tx.append(introw)
		else:
			y.append(int(row.pop()))
			introw = []
			for item in row:
				introw.append(float(item))
			x.append(introw)
		count += 1
	f.close()

	#these next lines reshape the arrays in order to return them in a useful state
	x = np.array(x)
	y = np.array(y)
	tx = np.array(tx).T
	ty = np.array(ty).reshape(np.array(ty).shape[0],1).T
	print("Test input shape",tx.shape)
	print("Test output shape",ty.shape)
	n_samples=len(y)
	print("Number of samples in the data set is :"+ str(n_samples))
	print("Shape of input matrix x is : "+str(x.shape))
	print("Shape of target vector y is :"+str(y.shape))
	
	X_train, y_train = x,y
	
	X_train=X_train.T
	y_train=y_train.reshape(y_train.shape[0],1)
	y_train=y_train.T
	
	Y_train_=np.zeros((outRange,y_train.shape[1]))
	for i in range(y_train.shape[1]):
		Y_train_[y_train[0,i],i]=1
	return X_train,y_train,Y_train_,tx,ty


	
	
