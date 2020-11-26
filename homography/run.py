import cv2 
import numpy as np
import matplotlib.pyplot as plt
from homography_utils import *

### Parameters 
fname = '1.jpg'

### Preparation 
folder = 'img/'

img = cv2.imread(folder+fname)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

if False: 
    cv2.imshow('image', img)
    cv2.waitKey(0)

### Running 
src_pts = np.array([
    [200, 1020], 
    [1884, 1050],
    [526, 768],
    [1703, 794]
])

dst_pts = np.array([
    [100, 1000], 
    [1000, 1000],
    [100, 100],
    [1000, 100]
])

M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
im_dst = cv2.warpPerspective(img, M, (1000, 1000))
print(M)

plt.subplot(1, 2, 1)
plt.imshow(img)
plt.subplot(1, 2, 2)
plt.imshow(im_dst)
# plt.show()

pts = np.concatenate([
        np.zeros((2, 1)), np.zeros((2, 1)) + 1
], axis=1)
print("Points: {}".format(pts))
print(find_close(pts, threshold=2))
