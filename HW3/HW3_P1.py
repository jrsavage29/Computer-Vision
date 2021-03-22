import numpy as np 
from numpy.linalg import inv
#Ax = N
A = np.empty
N = np.array([  [5], 
                [4], 
                [7], 
                [4], 
                [6], 
                [5] ])

x = np.array([  [0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 1], 
                [1, 0, 1, 0, 0, 0], 
                [0, 0, 0, 1, 0, 1], 
                [0, 1, 1, 0, 0, 0], 
                [0, 0, 0, 0, 1, 1] ])

A = np.dot( inv(x), N)

print(A)
print("Matrix A is of shape:" + str(A.shape) )