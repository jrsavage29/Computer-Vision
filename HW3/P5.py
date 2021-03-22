import numpy as np
import matplotlib.pyplot as plt
import cv2
import math
import random

#Calculate the geometric distance between estimated points and original points
def geometricDistance(correspondence, h):

    p1 = np.transpose(np.matrix([correspondence[0].item(0), correspondence[0].item(1), 1]))
    estimatep2 = np.dot(h, p1)
    estimatep2 = (1/estimatep2.item(2))*estimatep2

    p2 = np.transpose(np.matrix([correspondence[0].item(2), correspondence[0].item(3), 1]))
    error = p2 - estimatep2
    return np.linalg.norm(error)


def ransac(corr, iterations):
    maxInliers = []
    finalH = None
    for i in range(iterations):
        #find 4 random points to calculate a homography
        corr1 = corr[random.randrange(0, len(corr))]
        corr2 = corr[random.randrange(0, len(corr))]
        randomFour = np.vstack((corr1, corr2))
        corr3 = corr[random.randrange(0, len(corr))]
        randomFour = np.vstack((randomFour, corr3))
        corr4 = corr[random.randrange(0, len(corr))]
        randomFour = np.vstack((randomFour, corr4))

        #call the homography function on those points
        H = findHomography(randomFour)
        
        inliers = []

        for i in range(len(corr)):
            d = geometricDistance(corr[i], H)
            if d < 5:
                inliers.append(corr[i])

        if len(inliers) > len(maxInliers):
            maxInliers = inliers
            finalH = H
        #print ("Corr size: ", len(corr), " NumInliers: ", len(inliers), "Max inliers: ", len(maxInliers))

        threshold = 3
        if len(maxInliers) > (len(corr)*threshold):
            break
    return finalH, maxInliers

def findHomography(correspondences):
    #loop through correspondences and create assemble matrix
    aList = []
    for corr in correspondences:
        p1 = np.matrix([corr.item(0), corr.item(1), 1])
        p2 = np.matrix([corr.item(2), corr.item(3), 1])

        a1 = [-p2.item(2) * p1.item(0), -p2.item(2) * p1.item(1), -p2.item(2) * p1.item(2), 0, 0, 0,
              p2.item(0) * p1.item(0), p2.item(0) * p1.item(1), p2.item(0) * p1.item(2)]

        a2 = [0, 0, 0, -p2.item(2) * p1.item(0), -p2.item(2) * p1.item(1), -p2.item(2) * p1.item(2),
              p2.item(1) * p1.item(0), p2.item(1) * p1.item(1), p2.item(1) * p1.item(2)]
        
        aList.append(a1)
        aList.append(a2)

    matrixA = np.asarray(aList)

    #svd composition
    u, s, v = np.linalg.svd(matrixA)

    #reshape the min singular value into a 3 by 3 matrix
    H = np.reshape(v[8], (3, 3))

    #normalize and now we have h
    H = (1/H.item(8)) * H
    return H

def homography_warp(image, image2, H):
    height = len(image2)
    width = len(image2[0])
    new_height = len(image) + len(image2)
    new_width = len(image[0]) + len(image2[0])

    output = np.zeros((new_height, new_width, 3), dtype="uint8")
    
    for y_counter in range(0, len(image)):
        for x_counter in range(0, len(image[0])):
            curr_pixel = [x_counter,y_counter,1]

            if(curr_pixel[0] < height and curr_pixel[1] < width and curr_pixel[0] > 0 and curr_pixel[1] > 0):
                output[x_counter, y_counter] = image[curr_pixel[0], curr_pixel[1]]
    
    # H = np.linalg.inv(H)
    
    for y in range(output.shape[0]):
        for x in range(output.shape[1]):
            src = H.dot([x, y, 1])
            src = (src[:2] / src[2]).astype(int)

            if(0 <= src[0] < image2.shape[1]) and (0 <= src[1] < image2.shape[0]):
                val = image2[src[1], src[0], :]
                if y < image.shape[0] and x < image.shape[1]:

                    val = (val.astype(int) + image[y, x, :].astype(int)) / 2
                output[y, x, :] = val
            
    return output

def getCorrespodences(I1, I2):

    # Initialize the ORB detector algorithm 
    orb = cv2.ORB_create() 

    # Now detect the keypoints and compute 
    # the descriptors for the I1 image 
    # and I2 image 
    I1Keypoints, I1Descriptors = orb.detectAndCompute(I1,None) 
    I2Keypoints, I2Descriptors = orb.detectAndCompute(I2,None)

    # Initialize the Matcher for matching 
    # the keypoints and then match the 
    # keypoints 
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, True) 
    matches = matcher.match(I1Descriptors,I2Descriptors)
    matches = sorted(matches, key = lambda x:x.distance)
    matches = matches[:20]

    correspondenceList = []
    keypoints = [I1Keypoints, I2Keypoints]

    #print ('#matches:', len(matches))

    for match in matches:
        (x1, y1) = keypoints[0][match.queryIdx].pt
        (x2, y2) = keypoints[1][match.trainIdx].pt
        correspondenceList.append([x1, y1, x2, y2])

    corrs = np.matrix(correspondenceList)

    return corrs

#Output mosaic image is Iout
I1 = cv2.imread('wall1.png')
I2 = cv2.imread('wall2.png')

corrs1 = getCorrespodences(I1, I2)

iterations = 1000
#run ransac
finalH, maxInliers = ransac(corrs1, iterations)

#print("Max num of inliers: ", len(maxInliers))
#print("Final Homography with a shape of : ", finalH.shape)
#print(finalH)

Iout = homography_warp(I1, I2,finalH)


#Now for my image tests

myImg1 = cv2.imread('OutsideRight.png')
myImg2 = cv2.imread('OutsideMiddle.png')

corrs2 = getCorrespodences(myImg1, myImg2)

myH, maxInliers = ransac(corrs2, iterations)

Iout2 = homography_warp(myImg1, myImg2, myH)

#save the npy
np.save('outputP5H.npy', finalH)

cv2.imshow("The mosiac image", Iout)
cv2.imwrite('ouputP5wall.png', Iout)

cv2.imshow("Tested transformed image", Iout2)
cv2.imwrite('ouputP5myImages.png', Iout2)

cv2.waitKey(0)
cv2.destroyAllWindows()