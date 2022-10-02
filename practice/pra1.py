from re import I
import numpy as np
import matplotlib.pyplot as plt 
from functions import cross_entropy_error

t=[0,1,0,0,0,0]
x=[0.1,0.2,0.3,0.4,0,0]
y=[0.8,0,0.2,0,0,0]
a=cross_entropy_error(np.array(x),np.array(t))
b=cross_entropy_error(np.array(y),np.array(t))  
print(a)
print(b)