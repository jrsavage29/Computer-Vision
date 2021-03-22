import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from collections import Counter

#Create random 100 x 100 matrix A and store in inputP5A.npy file
A = np.random.randint(100, size = (100,100))
np.save('inputP5A.npy', A)

#Now load in the randomly generated matrix from the inputP5A.npy file
from_A = np.load('inputP5A.npy')


#############################5c 1.##############################################
x_vals = list(range(101))

#Create the sorted list of intensities from matrix A and plotting them
temp = from_A.flatten()
#temp.sort()
freq = Counter(temp)

y_vals = []

for i in range(len(x_vals)):
    y_vals.append(freq[i])

plt.rcParams["patch.force_edgecolor"] = True    
plt.figure(1)
plt.bar( x_vals, y_vals)
plt.title("Intensities of Each Element of Matrix A")
plt.xlabel("Elements of the Matrix A")
plt.ylabel("Intensity")

#########################5c 2.##########################################
plt.figure(2)
plt.hist(temp, bins = 20)
plt.title("Histogram of A")

#########################5c 3.##########################################
h = len(from_A)
w = len(from_A[1])

#Separate into quadrants
top_left  = [from_A[i][:50] for i in range(50)]
top_right = [from_A[i][50:] for i in range(50)]
bot_left =  [from_A[i][:50] for i in range(50, 100)]
bot_right = [from_A[i][50:] for i in range(50, 100)]

X = bot_left
plt.figure(3)
plt.imshow(X)
np.save('outputP5X.npy', X)

#########################5c 4.##########################################
mean_intensity = np.mean(list(temp))
Y = from_A - mean_intensity
plt.figure(4)
plt.imshow(Y)
np.save('outputP5Y.npy', Y)


#########################5c 5.##########################################
compare_list_of_A = temp
Z = []
for i in range(10000):
    if(mean_intensity > compare_list_of_A[i] ):
        Z.append([255, 0, 0])

    else:
        Z.append([0, 0, 0])

Z = np.reshape(Z, (100, 100, 3))
Z = Z.astype(np.uint8)

#print(Z)

img = Image.fromarray(Z)
img.save('outputP5Z.png')

plt.figure(5)
plt.imshow(Z)


plt.show()