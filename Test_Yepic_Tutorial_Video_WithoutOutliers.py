# -*- coding: utf-8 -*-
"""
Created on Sat Sep  7 13:58:14 2019

@author: YesAB
"""

import cv2
import numpy as np
from matplotlib import pyplot as plt
import os.path
import os
# =============================================================================
# script_path = os.path.dirname(os.path.abspath( __file__ ))
# parent_dir= os.path.abspath(os.path.join(script_path, os.pardir))
# print "file is --->:" + os.path.dirname(os.getcwd())
# parent_dir=os.path.dirname(os.getcwd())
# =============================================================================


# Get the working directory (to use a universal path anywhere)
wdir=os.getcwd()

# Original Code or retrieving images
# =============================================================================
# img1 = cv2.imread(wdir +"\img\src_02.png",cv2.IMREAD_GRAYSCALE)
# img2 = cv2.imread(wdir +"\img\\tar_02.png",cv2.IMREAD_GRAYSCALE) # Is this a trick (tar??)
# =============================================================================

# Trying colored images 
img1 = cv2.imread(wdir +"\img\src_02.png",0)
img2 = cv2.imread(wdir +"\img\\tar_02.png",0) # Is this a trick (tar??)

# Rescaling Images:
scale_percent = 40 # percent of original size

width = int(img1.shape[1] * scale_percent / 100)
height = int(img1.shape[0] * scale_percent / 100)
dim = (width, height)

# resize image 1
img1 = cv2.resize(img1, dim, interpolation = cv2.INTER_AREA)

# resize image 2
img2 = cv2.resize(img2, dim, interpolation = cv2.INTER_AREA)


#set a condition that atleast 10 matches have to be present
#Otherwise show a message saying not enough matches are present.
MIN_MATCH_COUNT = 10

# Initiate SIFT detector
sift = cv2.xfeatures2d.SIFT_create()

# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)

FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks = 50)

flann = cv2.FlannBasedMatcher(index_params, search_params)

matches = flann.knnMatch(des1,des2,k=2)

# store all the good matches as per Lowe's ratio test.
good = []
for m,n in matches:
    if m.distance < 0.7*n.distance:
        good.append(m)
        
if len(good)>MIN_MATCH_COUNT:
    src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
    matchesMask = mask.ravel().tolist()

#Drawing a border
# =============================================================================
#     h,w = img1.shape
#     pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
#     dst = cv2.perspectiveTransform(pts,M)
# 
#     img2 = cv2.polylines(img2,[np.int32(dst)],True,255,3, cv2.LINE_AA)
# =============================================================================

else:
    print "Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT)
    matchesMask = None
    
draw_params = dict(matchColor = None, 
                   singlePointColor = None,
                   matchesMask = matchesMask, # draw only inliers
                   flags = 2)


matching_result = cv2.drawMatches(img1,kp1,img2,kp2,good,None,**draw_params)
# =============================================================================
# for m in matches:
#     print(m.distance)
# 
# =============================================================================

#cv2.imshow("Image_1",img1)
#cv2.imshow("Image_2",img2)
print("Number of good matches: " + str(len(good)))
print "Matching_result: "
plt.figure(figsize=(15,15))
plt.imshow(matching_result)
plt.show()

# =============================================================================
# cv2.imshow(wdir +"\img\Matching_result", matching_result)                
# 
# cv2.waitKey(3500) # Wait for 3.5s and destroy
# cv2.destroyAllWindows()
# =============================================================================
