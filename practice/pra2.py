import numpy as np 
#パーセプトロンの勉強
def AND(x1,x2):
    x=np.array([x1,x2])
    w1,w2=0.5,0.5
    w=np.array([w1,w2])
    bias=-0.7
    threshold=np.sum(w*x)+bias
    if threshold<0:
        return 0
    else:
        return 1

def OR(x1,x2):
    x=np.array([x1,x2])
    w1,w2=0.5,0.5
    w=np.array([w1,w2])
    bias=-0.2
    threshold=np.sum(w*x)+bias
    if threshold<0:
        return 0
    else:
        return 1    

def NAND(x1,x2):
    x=np.array([x1,x2])
    w1,w2=0.5,0.5
    w=np.array([w1,w2])
    bias=-0.7
    threshold=np.sum(w*x)+bias
    if threshold>=0:
        return 0
    else:
        return 1    
    
def XOR(x1,x2):
    ans_NAND=NAND(x1,x2)
    ans_OR=OR(x1,x2)
    ans=AND(ans_NAND,ans_OR)
    return ans
        
print(XOR(0,0))
print(XOR(1,0))
print(XOR(0,1))
print(XOR(1,1))        