import numpy as np
import cv2
import time
from csv import reader
import os
import sys
sys.path.append('/home/pi/Documents/New documents/ai/deep_neural_network/myDigits')
np.set_printoptions(linewidth=np.inf)

n_layers = 3


def getQD(size):
	if size=="120x80":
		qd = 4
	elif size=="40x27":
		qd = 12
	return qd

def shrinkData(inp,size="120x80"):#take in normal image
	qd = getQD(size)
	out = cv2.resize(  cv2.cvtColor(  inp[0:480,160:480]  ,cv2.COLOR_BGR2GRAY)  ,(0,0),fx=1/qd,fy=1/qd)
	out = np.uint8(out>127)
	return out #output low data version
	
def readParams(size="120x80"):
	layers = {}
	f = open("myDigitReaderParameters_"+size+".txt","r")
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

def readDigits(size="120x80",test=0,outRange=10):
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
			number = count%(int(1/test))
		except:
			number = 1#anything except 0
		if number==0:
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
	x = np.array(x)
	y = np.array(y)
	tx = np.array(tx).T
	ty = np.array(ty).reshape(np.array(ty).shape[0],1).T
	print(tx.shape)
	print(ty.shape)
	n_samples=len(y)
	print("Number of samples in the data set is :"+ str(n_samples))
	print("Shape of input matrix x is : "+str(x.shape))
	print("Shape of target vector y is :"+str(y.shape))
	
	X_train, y_train = x,y
	
	X_train=X_train.T
	y_train=y_train.reshape(y_train.shape[0],1)
	y_train=y_train.T
	
	#print(y_train.shape)
	#print(outRange)
	#print(y_train)
	
	Y_train_=np.zeros((outRange,y_train.shape[1]))
	#print(Y_train_)
	for i in range(y_train.shape[1]):
		Y_train_[y_train[0,i],i]=1
	return X_train,y_train,Y_train_,tx,ty

def thing():
	lines = []
	f = open("myDigits_120x80.csv", 'r')
	csv_reader = reader(f)
	for row in csv_reader:
		line = ""
		if not row:
			continue
		y = int(row.pop())
		for item in row:
			line = line+str(operation(float(item)))+", "
		line = line+str(y)
		lines.append(line)
	f.close()
	f = open("myDigits_120x80.csv", 'w')
	for line in lines:
		f.write(line+"\n")
	f.close()

	
	
