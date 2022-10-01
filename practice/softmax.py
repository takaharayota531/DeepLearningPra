import numpy as np

def softmax_myself(array_a):
    max_value=np.max(array_a)
    array_a = array_a-max_value

    return np.exp(array_a)/np.sum(np.exp(array_a))

a=np.array([0.3,2.9,4.0])
y=softmax_myself(a)
print(y)
