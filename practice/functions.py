import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1/(1+np.exp(-x))
def relu(x):    
    return np.maximum(0,x)
def softmax(a):
    c=np.max(a)
    exp_a=np.exp(a-c)
    sum_exp_a=np.sum(exp_a)
    return exp_a/sum_exp_a
def sum_squared_error(y,t):
    return 1/2*(y-t)**2
def cross_entropy_error(y,t):
    delta=1e-7
    return -np.sum(t*np.log(y+delta))    
def numerical_diff(f,x):
    h=1e-4
    return (f(x+h)-f(x-h))/(2*h)
def numerical_gradient(f,x):
    h=1e-4
    grad=np.zeros_like(x)
    
    for idx in range(x.size):
        tmp_val=x[idx]
        x[idx]=tmp_val+h
        fxh1=f(x)
        x[idx]=tmp_val-h
        fxh2=f(x)
        grad[idx]=(fxh1-fxh2)/(2*h)
        x[idx]=tmp_val
    return grad
def gradient_descent(f, init_x, lr=0.01, step_num=100):
    x = init_x
    x_history = []

    for i in range(step_num):
        x_history.append( x.copy() )

        grad = numerical_gradient(f, x)
        x -= lr * grad

    return x
def function_2(x):
    if x.ndim == 1:
        return np.sum(x**2)
    else:
        return np.sum(x**2, axis=1)    
        
        