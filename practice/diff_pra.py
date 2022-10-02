import numpy as np
from functions import numerical_diff,function_2,numerical_gradient,gradient_descent
print(numerical_gradient(function_2,np.array([0.0,2.0])))#少数じゃないと丸めこまれてしまう

init_x=np.array([-3.0,4.0])
print(gradient_descent(function_2,init_x=init_x,lr=0.1,step_num=100))
print(gradient_descent(function_2,init_x=init_x,lr=10.0,step_num=100))
print(gradient_descent(function_2,init_x=init_x,lr=1e-10,step_num=100))