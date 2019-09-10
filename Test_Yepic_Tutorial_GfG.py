# -*- coding: utf-8 -*-
"""
Created on Sat Sep  7 13:52:36 2019

@author: YesAB
"""

import cv2 
import numpy as np 
from matplotlib import pyplot as plt
import os.path
import os

# Open the image files. 
wdir=os.getcwd()
# =============================================================================
# print(" The directory is --> " + wdir)
# print(" The file is --> " + wdir +"\img\src_02.png")
print(" The file is --> " + wdir +"\img\\tar_02.png")
# =============================================================================

img2_color = cv2.imread(wdir +"\img\src_02.png")
img1_color = cv2.imread(wdir +"\img\\tar_02.png")
  
# Open the image files. 
# =============================================================================
# img1_color = cv2.imread("tar_02.png")  # Image to be aligned. 
# img2_color = cv2.imread("src_02.png")    # Reference image. 
# =============================================================================

# Rescaling Images:
scale_percent = 40 # percent of original size

width = int(img1_color.shape[1] * scale_percent / 100)
height = int(img1_color.shape[0] * scale_percent / 100)
dim = (width, height)
# resize image 1
img1_color = cv2.resize(img1_color, dim, interpolation = cv2.INTER_AREA)

# resize image 2
img2_color = cv2.resize(img2_color, dim, interpolation = cv2.INTER_AREA)
  
# Convert to grayscale. 
img1 = cv2.cvtColor(img1_color, cv2.COLOR_BGR2GRAY) 
img2 = cv2.cvtColor(img2_color, cv2.COLOR_BGR2GRAY) 
height, width = img2.shape 
  
# Create ORB detector with 5000 features. 
orb_detector = cv2.ORB_create(5000) 
  
# Find keypoints and descriptors. 
# The first arg is the image, second arg is the mask 
#  (which is not reqiured in this case). 
kp1, d1 = orb_detector.detectAndCompute(img1, None) 
kp2, d2 = orb_detector.detectAndCompute(img2, None) 
  
# Match features between the two images. 
# We create a Brute Force matcher with  
# Hamming distance as measurement mode. 
matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = True) 
  
# Match the two sets of descriptors. 
matches = matcher.match(d1, d2) 
  
# Sort matches on the basis of their Hamming distance. 
matches.sort(key = lambda x: x.distance) 
  
# Take the top 90 % matches forward. 
matches = matches[:int(len(matches)*90)] 
no_of_matches = len(matches) 
  
# Define empty matrices of shape no_of_matches * 2. 
p1 = np.zeros((no_of_matches, 2)) 
p2 = np.zeros((no_of_matches, 2)) 
  
for i in range(len(matches)): 
  p1[i, :] = kp1[matches[i].queryIdx].pt 
  p2[i, :] = kp2[matches[i].trainIdx].pt 
  
# Find the homography matrix. 
homography, mask = cv2.findHomography(p1, p2, cv2.RANSAC) 
  
# Use this matrix to transform the 
# colored image wrt the reference image. 
transformed_img = cv2.warpPerspective(img1_color, 
                    homography, (width, height)) 
  
# Save the output. 
#cv2.imwrite('output.png', transformed_img) 
#cv2.imshow('output.png', transformed_img) 

print "transformed_img: "
plt.figure(figsize=(15,15))
plt.imshow(transformed_img)
plt.show()

# =============================================================================
# cv2.waitKey(3500) # Wait for 3.5s and destroy
# cv2.destroyAllWindows()
# 
# =============================================================================
