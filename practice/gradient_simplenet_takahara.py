import sys,os
sys.path.append(os.pardir)
import numpy as np

from functions import softmax,cross_entropy_error,numerical_gradient


class simpleNet: