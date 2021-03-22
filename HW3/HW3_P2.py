import numpy as np 
from numpy.linalg import inv

x1_p = 5
y1_p = 4
x2_p = 7
y2_p = 4
x3_p = 7
y3_p = 5
x4_p = 6
y4_p = 6

x1 = 0
y1 = 0
x2 = 1
y2 = 0
x3 = 1
y3 = 1
x4 = 0
y4 = 1
#Ax = N
A = np.empty
N = np.array([  [x1_p], 
                [y1_p], 
                [x2_p], 
                [y2_p], 
                [x3_p], 
                [y3_p],
                [x4_p],
                [y4_p],])

x = np.array([  [x1, y1, 1, 0, 0, 0, -x1_p*x1, -x1_p*y1],
                [0, 0, 0, x1, y1, 1, -y1_p*x1, -y1_p*y1], 
                [x2, y2, 1, 0, 0, 0, -x2_p*x2, -x2_p*y2],
                [0, 0, 0, x2, y2, 1, -y2_p*x2, -y2_p*y2], 
                [x3, y3, 1, 0, 0, 0, -x3_p*x3, -x3_p*y3],
                [0, 0, 0, x3, y3, 1, -y3_p*x3, -y3_p*y3],
                [x4, y4, 1, 0, 0, 0, -x4_p*x4, -x4_p*y4],
                [0, 0, 0, x4, y4, 1, -y4_p*x4, -y4_p*y4], ])

A = np.dot( inv(x), N)

print(A)
print("Matrix A is of shape:" + str(A.shape) )