import numpy as np
import cv2
from homography_utils import *

### Testing Method correctness 
src = np.array([[1], [3]])
dst = np.array([[5], [7]])

print(compute_homography_build_A(src, dst))

src = np.random.rand(2, 4)
dst = np.random.rand(2, 4)
print(compute_homography(src, dst))

### Testing against cv2
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
norm = np.sum(M * M) ** (1/2)

print("Normalized CV2:\n{}".format(M / norm))
print("Monroe-Penrose:\n{}".format(compute_homography(src_pts.T, dst_pts.T)))
