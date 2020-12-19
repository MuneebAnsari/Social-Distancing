import cv2 
import numpy as np
import matplotlib.pyplot as plt
from homography_utils import *

### Parameters 
fname = '1.jpg'

### Preparation 
folder = 'img/'
boxes = [((676, 688), (759, 915)), ((375, 779), (484, 1092)), ((761, 694), (840, 913)), ((219, 818), (335, 1074)), ((1450, 636), (1525, 829)), ((1629, 678), (1704, 870)), ((971, 615), (1036, 809)), ((1040, 613), (1116, 812)), ((874, 545), (928, 680)), ((1668, 823), (1766, 1077)), ((1182, 621), (1236, 803)), ((923, 549), (973, 675)), ((1019, 570), (1068, 660)), ((815, 549), (854, 709))] # The bounding boxes for pedestrians obtained from darknet. 

img = cv2.imread(folder+fname)
# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

##### Runs 
### Obtaining the figure with pedestrians in boxes
detect_img = color_boxes(img, boxes)
cv2.imwrite("homography_1.jpg", detect_img)

### Obtaining the figure with points below pedestrians
pts = box2pt(boxes) # bounding box to point at bottom center
detect_pt_img = color_pts(np.copy(detect_img), pts, color=[0, 0, 255])
cv2.imwrite("homography_2.jpg", detect_pt_img)

### Compute homography
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
H = compute_homography(src_pts.T, dst_pts.T)

### Compute transformed pedestrians coordinates 
pts_T = np.array(pts).T
pts_trsf = homography_trsf(pts_T, H) 

### Find the indices of close pedestrians 
cls_idx = find_close(pts_trsf, threshold=100, verbose=False)
print(cls_idx)

### Plotting the close boxes in red
close_img = np.copy(detect_img)
for idx in cls_idx:
  box = boxes[idx]
  cv2.rectangle(close_img, box[0], box[1], [0, 0, 255], 3)

cv2.imwrite("homography_3.jpg", close_img)

### Find the indices of close pedestrians, with idx=12 being the one without mask
cls_idx = find_close(pts_trsf, threshold=100, verbose=False, idx_stay_away=[12])
print(cls_idx)

### Plotting the close boxes in red
close_img = np.copy(detect_img)
for idx in cls_idx:
  box = boxes[idx]
  cv2.rectangle(close_img, box[0], box[1], [0, 0, 255], 3)

cv2.imwrite("homography_4.jpg", close_img)