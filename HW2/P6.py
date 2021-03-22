import numpy as np
import matplotlib.pyplot as plt
import cv2
import math

def affine_warp(A, input_image):
    #takes an image and matrices and applies it.  
    x_min = 0
    y_min = 0
    x_max = input_image.shape[0]
    y_max = input_image.shape[1]

    res_image = np.zeros((x_max, y_max), dtype= "uint8")

    for y_counter in range(0, y_max):
        for x_counter in range(0, x_max):
            curr_pixel = [x_counter,y_counter,1]

            curr_pixel = np.dot(A, curr_pixel)

            # print(curr_pixel)

            if curr_pixel[0] > x_max - 1 or curr_pixel[1] > y_max - 1 or x_min > curr_pixel[0] or y_min > curr_pixel[1]:
                next
            else:
                res_image[x_counter][y_counter] = input_image[int(curr_pixel[0])][int(curr_pixel[1])] 

    return res_image


#Load an image in grayscale
img = cv2.imread('boat.png', 0)

rotate_A = np.array([[ math.cos( math.radians(-10) ), -(math.sin( math.radians(-10) )), 0 ], [ math.sin( math.radians(-10) ), math.cos( math.radians(-10) ), 0 ], [ 0, 0, 1 ]])
translate_A = np.array([ [1, 0, -90], [0, 1, 120], [0, 0, 1] ])

multiplied_matrices = np.dot(rotate_A, translate_A)
inverse_transform_matrix = np.linalg.inv(multiplied_matrices)

plt.figure("Grayscale Image")
plt.imshow(img, cmap = 'Greys_r')

plt.figure("Transformed Image")

transformed_img = affine_warp(inverse_transform_matrix, img)
plt.imshow(transformed_img, cmap='Greys_r')
plt.imsave('outputP6.png', transformed_img, cmap = 'Greys_r')
np.save("outputP6A.npy", inverse_transform_matrix)

plt.show()