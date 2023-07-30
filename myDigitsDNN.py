#import numpy as np
from numpy import set_printoptions, inf, random, dot ,zeros, divide, clip, array2string, exp, round, sum, squeeze, argmax, log
set_printoptions(linewidth=inf)
import matplotlib.pyplot as plt
from csv import reader
import myDigitsReaderHelp as hlp
#import time
from time import time, sleep
import datetime

size = "40x27"

testEnabled = True
X_train,y_train,Y_train_,x_test,y_test = hlp.readDigits(size=size,outRange=10,test=0.1)###############################################################################################

if size == "120x80":
	layers_dims=[9600,120,40,4]###########################################################################################################################################
elif size == "40x27":
	layers_dims=[1080,120,40,10]###########################################################################################################################################

def initialize_parameters_deep(layer_dims):
	random.seed(3)
	parameters = {}
	L = len(layer_dims) 
	for l in range(1, L):
		parameters['W' + str(l)] = random.randn(layer_dims[l],layer_dims[l-1])*0.01
		parameters['b' + str(l)] = zeros((layer_dims[l],1))
		
		assert(parameters['W' + str(l)].shape == (layer_dims[l], layer_dims[l-1]))
		assert(parameters['b' + str(l)].shape == (layer_dims[l], 1))
	return parameters

def linear_forward(A, W, b):
	Z = dot(W,A)+b
	assert(Z.shape == (W.shape[0], A.shape[1]))
	cache = (A, W, b)
	
	return Z, cache

def sigmoid_(Z):
	return 1/(1+exp(-Z))

def relu_(Z):
	return Z*(Z>0)

def drelu_(Z):
	return 1. *(Z>0)

def dsigmoid_(Z):
	return sigmoid_(Z)*(1-sigmoid_(Z))

def sigmoid(Z):
	return sigmoid_(Z),Z

def relu(Z):
	return relu_(Z),Z

def linear_activation_forward(A_prev,W,b,activation):
	if activation == "sigmoid":
		Z, linear_cache = linear_forward(A_prev,W,b)
		A, activation_cache = sigmoid(clip(Z,-25,25))
		
	elif activation == "relu":
		Z, linear_cache = linear_forward(A_prev,W,b)
		A, activation_cache = relu(Z)
		
	assert (A.shape == (W.shape[0], A_prev.shape[1]))
	cache = (linear_cache, activation_cache)
	
	return A, cache

def L_model_forward(X, parameters):
	caches = []
	A = X
	L = len(parameters) // 2
	for l in range(1, L):
		A_prev = A
		A, cache = linear_activation_forward(A_prev,parameters['W'+str(l)],parameters['b'+str(l)],"relu")
		caches.append(cache)
	AL, cache = linear_activation_forward(A,parameters['W'+str(L)],parameters['b'+str(L)],"sigmoid")
	caches.append(cache)
	return AL, caches

def compute_cost(AL, Y):
	m=Y.shape[1]
	cost = -(1/m)*sum((Y*log(AL)+(1-Y)*log(1-AL)))
	#cost = sum((Y-AL)**2)/m
	cost=squeeze(cost)
	assert(cost.shape == ())
	return cost

def linear_backward(dZ, cache):
	A_prev, W, b = cache
	m = A_prev.shape[1]
	dW = (1/m)*dot(dZ,A_prev.T)
	db = (1/m)*sum(dZ,axis=1,keepdims=True)
	dA_prev = dot(W.T,dZ)
	
	assert (dA_prev.shape == A_prev.shape)
	assert (dW.shape == W.shape)
	assert (db.shape == b.shape)
	
	return dA_prev, dW, db

def relu_backward(dA,activation_cache):
	return dA* drelu_(activation_cache)

def sigmoid_backward(dA,activation_cache):
	return dA* dsigmoid_(activation_cache)

def linear_activation_backward(dA, cache, activation):
	linear_cache, activation_cache = cache
	if activation == "relu":
		dZ = relu_backward(dA,activation_cache)
		dA_prev, dW, db = linear_backward(dZ,linear_cache)
	
	elif activation == "sigmoid":
		dZ = sigmoid_backward(dA,activation_cache)
		dA_prev, dW, db = linear_backward(dZ,linear_cache)
	return dA_prev,dW,db

def L_model_backward(AL, Y, caches):
	grads = {}
	L = len(caches)
	m = AL.shape[1]
	dAL = - (divide(Y, AL) - divide(1 - Y, 1 - AL))
	
	current_cache = caches[L-1]
	grads["dA" + str(L-1)], grads["dW" + str(L)], grads["db" + str(L)] = linear_activation_backward(dAL,current_cache,"sigmoid")
	
	for l in reversed(range(L-1)):
		current_cache = caches[l]
		dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads["dA"+str(l+1)],current_cache,"relu")
		grads["dA" + str(l)] = dA_prev_temp
		grads["dW" + str(l + 1)] = dW_temp
		grads["db" + str(l + 1)] = db_temp
	return grads

def update_parameters(parameters, grads, learning_rate):
	L = len(parameters) // 2 
	for l in range(L):
		parameters["W" + str(l+1)] = parameters["W" + str(l+1)]-(learning_rate)*grads["dW"+str(l+1)] 
		parameters["b" + str(l+1)] = parameters["b" + str(l+1)]-(learning_rate)*grads["db"+str(l+1)]
	return parameters

def predict_L_layer(X,parameters,out=False):
	AL,caches=L_model_forward(X,parameters)
	if out:
		print(round(AL,3))
	prediction=argmax(AL,axis=0)
	return prediction.reshape(1,prediction.shape[0])

def L_layer_model(X, Y, layers_dims, learning_rate = 0.01, num_iterations = 3000, print_cost=False):
	random.seed(1)
	costs = []
	fails = []
	
	usenew = input("Would you like to start from scratch?>>> ")
	if "y" in usenew.lower():
		parameters = initialize_parameters_deep(layers_dims)
	else:
		parameters = hlp.readParams(size=size)
	
	ptime = time()
	for i in range(0, num_iterations):
		AL, caches = L_model_forward(X, parameters)
		grads = L_model_backward(AL, Y, caches)
		parameters = update_parameters(parameters, grads, learning_rate)
		if print_cost and i % 50 == 0:
			cost = compute_cost(AL, Y)
			if i>0:
				spd = round(50/(time()-ptime),2)
			else:
				spd = round(1/(time()-ptime),2)
			if testEnabled:
				fail = sum(predict_L_layer(x_test,parameters)!=y_test)/y_test.shape[1] * 100
				fails.append(fail)
				fail = str(round(fail,2))+"%"
			else:
				fail = "Disabled"
			predTime = (num_iterations-i)/spd
			print("\r\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\r",end="",flush=True)
			print("Iteration: "+str(i)+"/"+str(num_iterations-1)+"  Cost: "+str(round(cost,3))+
				"  Fail rate: "+fail+"  Speed: "+str(spd)+"its/s  Predicted finish: "+
				str((datetime.datetime.now()+datetime.timedelta(0,predTime)).time().strftime("%X")),end="",flush=True)
			ptime = time()
			costs.append(cost)
	print()
	plt.plot(squeeze(costs))
	plt.ylabel('cost')
	plt.xlabel('iterations (per hundreds)')
	plt.title("Learning rate =" + str(learning_rate))
	plt.show()
	
	if testEnabled:
		plt.plot(squeeze(fails))
		plt.ylabel('Percentage of testing data that failed')
		plt.xlabel('iterations (per hundreds)')
		plt.show()
	return parameters

parameters = L_layer_model(X_train, Y_train_, layers_dims, num_iterations = 50001, print_cost = True)

#Test:
predictions_train_L = predict_L_layer(X_train, parameters)
print("Training Accuracy : "+ str(sum(predictions_train_L==y_train)/y_train.shape[1] * 100)+" %")
save = input("Would you like to save these parameters?>>> ")
if "y" in save.lower():
	filename = "myDigitReaderParameters_"+size+".txt"
	f = open(filename,"w")
	for l in range(1,len(layers_dims)):
		arr = parameters["W"+str(l)]
		string = array2string(arr,threshold=110000000)
		string = string.replace("\n",", ")
		f.write(string+"\n")
		arr = parameters["b"+str(l)]
		string = array2string(arr,threshold=110000000)
		string = string.replace("\n",", ")
		f.write(string+"\n")
	f.close()
	print("Saved in "+filename)
print("Goodbye")
	
