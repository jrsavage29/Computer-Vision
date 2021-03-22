import numpy as np

y = np.array([1, 2, 3, 4, 5, 6])
#print(y)

z = y.reshape(3,2)
#print(z)

r = np.where( z == np.max(z) )[0]
#print(r)
c = np.where( z == np.max(z) )[1]
#print(c)
x = np.max(z)
#print(x)

v = np.array([1, 8, 8, 2, 1, 3, 9, 8])
x = np.count_nonzero(v == 1)
#print(x)

dice = np.random.randint(1,7, size = 12)
#print(dice)