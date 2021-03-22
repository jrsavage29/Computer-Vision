import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image

def stereo_match_SSD(left_img, right_img, kernel, max_offset):
    # Load in both images, assumed to be RGBA 8bit per channel images, convert to gray scale
    left_img = Image.open(left_img).convert('L')
    left = np.asarray(left_img)
    right_img = Image.open(right_img).convert('L')
    right = np.asarray(right_img)    
    w, h = left_img.size  # assume that both images are same size   
    
    # Depth (or disparity) map
    depth = np.zeros((w, h), np.uint8)
    depth.shape = h, w
       
    kernel_half = int(kernel / 2)    
    offset_adjust = 255 / max_offset  # this is used to map depth map output to 0-255 range
      
    for y in range(kernel_half, h - kernel_half):      
        print(".", end="", flush=True)  # let the me know that something is happening 
        
        for x in range(kernel_half, w - kernel_half):
            best_offset = 0
            prev_ssd = 65534
            
            for offset in range(max_offset):               
                ssd = 0
                ssd_temp = 0                            
                
                # v and u are the x,y of our local window search, used to ensure a good 
                # match- going by the squared differences of two pixels alone is insufficient, 
                # we want to go by the squared differences of the neighbouring pixels too
                for v in range(-kernel_half, kernel_half):
                    for u in range(-kernel_half, kernel_half):
                        # iteratively sum the sum of squared differences value for this block
                        # left[] and right[] are arrays of uint8, so converting them to int saves
                        # potential overflow, and executes a lot faster 
                        ssd_temp = int(left[y+v, x+u]) - int(right[y+v, (x+u) - offset])  
                        ssd += ssd_temp * ssd_temp              
                
                # if this value is smaller than the previous ssd at this block
                # then it's theoretically a closer match. Store this value against
                # this block..
                if ssd < prev_ssd:
                    prev_ssd = ssd
                    best_offset = offset
                            
            # set depth output for this x,y location to the best match
            depth[y, x] = best_offset * offset_adjust
                                
    # Convert to PIL and save it
    Image.fromarray(depth).save('disp_map.png')
    cv2.imshow("The Disparity Map with SSD", depth)

def stereo_match_NCC(left_img, right_img, kernel, max_offset):
    # Load in both images, assumed to be RGBA 8bit per channel images, convert to gray scale
    left_img = Image.open(left_img).convert('L')
    left = np.asarray(left_img)
    right_img = Image.open(right_img).convert('L')
    right = np.asarray(right_img)    
    w, h = left_img.size  # assume that both images are same size   
    
    # Depth (or disparity) map
    depth = np.zeros((w, h), np.uint8)
    depth.shape = h, w
       
    kernel_half = int(kernel / 2)    
    offset_adjust = 255 / max_offset  # this is used to map depth map output to 0-255 range
      
    for y in range(kernel_half, h - kernel_half):      
        print(".", end="", flush=True)  # let the me know that something is happening 
        
        for x in range(kernel_half, w - kernel_half):
            best_offset = 0
            prev_ncc = 65534
            
            for offset in range(max_offset):               
                ncc = 0
                l_mean = 0
                r_mean = 0
                n = 0                           
                
                # v and u are the x,y of our local window search, used to ensure a good 
                # match- going by the normalized cross correlation of two pixels alone is insufficient, 
                # we want to go by the normalized cross correlation of the neighbouring pixels too
                for v in range(-kernel_half, kernel_half):
                    for u in range(-kernel_half, kernel_half):
                        #calculate the cumulative sum
                        l_mean += left[y+v, x+u]
                        r_mean += right[y+v, (x+u) - offset] 
                        n += 1              
                
                l_mean = l_mean/n
                r_mean = r_mean/n

                l_r = 0
                l_var = 0
                r_var = 0

                for v in range(-kernel_half, kernel_half):
                    for u in range(-kernel_half, kernel_half):
                        
                        l = int(left[y+v, x+u]) - l_mean
                        r = int(right[y+v, (x+u) - offset]) - r_mean
                        
                        l_r += l*r
                        l_var += l**2
                        r_var += r**2

                        ncc = -l_r/np.sqrt(l_var*r_var)


                # if this value is smaller than the previous ncc at this block
                # then it's theoretically a closer match. Store this value against
                # this block..
                if ncc < prev_ncc:
                    prev_ncc = ncc
                    best_offset = offset
                            
            # set depth output for this x,y location to the best match
            depth[y, x] = best_offset * offset_adjust
                                
    # Convert to PIL and save it
    Image.fromarray(depth).save('disp_map_NCC.png')
    cv2.imshow("The Disparity Map with NCC", depth)


if __name__ == '__main__':
    stereo_match_SSD("image_left.png", "image_right.png", 6, 30)  # 6x6 local search kernel, 30 pixel search range

    #Uncomment bottom line if you want to run stereo matching with NCC instead
    #stereo_match_NCC("image_left.png", "image_right.png", 6, 30)  # 6x6 local search kernel, 30 pixel search range

    cv2.waitKey(0)
    cv2.destroyAllWindows()