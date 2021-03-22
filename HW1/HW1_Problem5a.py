import numpy as np #importing the numpy module into the short name np

a = np.array([[1,2,3], [4,5,6], [7,8,9]]) #creates a 3x3 matrix
b = a[2,:] #since python starts from a zero index, this will set be equal to the 3rd row of the a matix
c = a.reshape(-1) #c will contain a single row of all the unchanged elements of matrix a because we specified an unknown
                    #number of rows
f = np.random.randn(5,1) #create a 5x1 matrix filled with random values as per standard normal distribution.
g = f[f>0] #g will only be a 3x1 matrix because it will only store values greater than 0 from f's matrix
x = np.zeros(10)+0.5 #x will be an array of 10 "padded" zeroes that also have 0.5 added to each element (10 entries
                        # of 0.5) 
y = 0.5*np.ones(len(x)) #y will be an array the same length as x but padded with 1s instead. Each element is then
                        # multiplied by 0.5
z = x + y               #matrix addition of x and y are stored in z (An array of 10 1s)
a = np.arange(1,100) #creates an array of values from 1 to 99 with even spacing with the default step size of 1
b = a[::-1] #Starting from the back of the array a, copy all values into b
c = np.random.permutation(10) #creates a randomly permuted array with a range of 10

