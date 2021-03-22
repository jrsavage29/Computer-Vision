import numpy as np
import matplotlib.pyplot as plt
import cv2
import scipy.stats as st

def dnorm(x, y, sd):
    return 1 / (np.sqrt(2 * np.pi) * sd) * np.e ** (-np.power((x - y) / sd, 2) / 2)


def gauss_kern(kernlen, sigma):
    """Returns a 2D Gaussian kernel."""
    kern_1D = np.linspace(-(kernlen // 2), kernlen // 2, kernlen )
    for i in range(kernlen):
        kern_1D[i] = dnorm(kern_1D[i], 0, sigma)
    kern_2D = np.outer(kern_1D.T, kern_1D.T)
    kern_2D *= 1.0 / kern_2D.max()

    return kern_2D


#Load an image in grayscale
img = cv2.imread('inputP6.jpg', cv2.IMREAD_GRAYSCALE)

plt.figure("Before Gaussian Filtering")
plt.imshow(img, cmap = 'Greys_r')
#print(img.shape)


ksize = 5
denominator = 273.0

#apply a kernel
kernel =   gauss_kern(ksize, 1.414)*denominator
#print(kernel)

image_row, image_col = img.shape
kernel_row, kernel_col = kernel.shape

#create an output image
res = np.zeros(img.shape)
 
pad_height = int((kernel_row - 1) / 2)
pad_width = int((kernel_col - 1) / 2)
 
padded_image = np.zeros((image_row + (2 * pad_height), image_col + (2 * pad_width)))
 
padded_image[pad_height:padded_image.shape[0] - pad_height, pad_width:padded_image.shape[1] - pad_width] = img
 
 
for row in range(image_row):
    for col in range(image_col):
        res[row, col] = np.sum(kernel * padded_image[row:row + kernel_row, col:col + kernel_col])

plt.figure("After Gaussian Filtering") 
plt.imshow(res, cmap = 'Greys_r')
plt.title("Output Image using {}X{} Kernel".format(kernel_row, kernel_col))
#print(res.shape)

plt.show()
 

plt.imsave('outputP6.png', res, cmap = 'Greys_r')
#image = Image.fromarray(res)
#image.save('outputP66.png')