import numpy as np
import matplotlib.pyplot as plt
import cv2

def sobel_edge_detection(img, sobel_filter):
    image_row, image_col = img.shape
    kernel_row, kernel_col = sobel_filter.shape
    res_x = np.zeros(img.shape)
    res_y = np.zeros(img.shape)
    res = np.zeros(img.shape)

    sobel_filter_x = sobel_filter

    sobel_filter_y = sobel_filter.transpose()

    # print(sobel_filter_x)
    # print(sobel_filter_y)

    pad_height = int((kernel_row - 1) / 2)
    pad_width = int((kernel_col - 1) / 2)
    
    padded_image = np.zeros((image_row + (2 * pad_height), image_col + (2 * pad_width)))
    
    padded_image[pad_height:padded_image.shape[0] - pad_height, pad_width:padded_image.shape[1] - pad_width] = img
    
    
    for row in range(image_row):
        for col in range(image_col):
            res_x[row, col] = np.sum(sobel_filter_x * padded_image[row:row + kernel_row, col:col + kernel_col])

    for row in range(image_row):
        for col in range(image_col):
            res_y[row, col] = np.sum(sobel_filter_y * padded_image[row:row + kernel_row, col:col + kernel_col])

    
    gradient_magnitude = np.sqrt( np.square(res_x) + np.square(res_y) )
    gradient_magnitude *= 255/ gradient_magnitude.max()
    #print(gradient_magnitude)
    res = np.array(gradient_magnitude)
    plt.figure("After Sobel Edge Detection")
    plt.imshow(res, cmap = 'Greys_r')
    #cv2.imshow('Sobel Operator Result', gradient_magnitude)
    plt.imsave('outputP5sobel.png', res, cmap = 'Greys_r')
    

#Load an image in grayscale
img = cv2.imread('boat.png', 0)
plt.figure("Grayscale Image")
plt.imshow(img, cmap = 'Greys_r')
#cv2.imshow('Grayscale Image',img)


sobel_filter = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])

sobel_edge_detection(img, sobel_filter)

#The source I got the following code from to make a more accurate
# canny edge detection result https://www.pyimagesearch.com/2015/04/06/zero-parameter-automatic-canny-edge-detection-with-python-and-opencv/

# compute the median of the single channel pixel intensities
sigma = 0.33
v = np.median(img)
# apply automatic Canny edge detection using the computed median
lower = int(max(0, (1.0 - sigma) * v))
upper = int(min(255, (1.0 + sigma) * v))
edges = cv2.Canny(img, lower, upper)

plt.figure("After Canny Edge Detection")
plt.imshow(edges, cmap = 'Greys_r')
#cv2.imshow('Sobel Operator Result', gradient_magnitude)
plt.imsave('outputP5canny.png', edges, cmap = 'Greys_r')

plt.show()
cv2.waitKey(0)
cv2.destroyAllWindows()