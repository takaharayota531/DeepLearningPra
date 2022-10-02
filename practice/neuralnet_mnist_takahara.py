#coding: utf-8
import sys,os
sys.path.append(os.pardir)
import numpy as np
import pickle
from dataset.mnist import load_mnist
from functions import sigmoid,softmax
import time

def get_data():
    (x_train, t_train),(x_test, t_test) = load_mnist(normalize=True,flatten=True,one_hot_label=False)
    return x_test, t_test
def init_network():
    with open("sample_weight.pkl",'rb') as f:
        network=pickle.load(f)
    return network
def predict(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = softmax(a3)

    return y

x,t=get_data()
print(x.shape)
network=init_network()
accuracy_count=0
start_time=time.perf_counter()
batch_size=100
for i in range(0,len(x),batch_size):
    x_batch=x[i:i+batch_size]
    y_batch=predict(network,x_batch) 
    largest_index=np.argmax(y_batch,axis=1)
    accuracy_count+=np.sum(largest_index==t[i:i+batch_size])
finish_time=time.perf_counter()  
print("Accuracy"+str(float(accuracy_count)/len(x)))   
print("計測時間"+str(finish_time-start_time))