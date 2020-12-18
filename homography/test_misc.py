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

### Testing coloring boxes & pts 
boxes = [((100, 200), (300, 400)), ((500, 500), (600, 600))]
pts = box2pt(boxes)
print(pts)

img = color_boxes(img, boxes)
img = color_pts(img, pts, color=[255, 0, 0])

plt.imshow(img)
plt.show()

### Testing homography transformation

pts = np.array([[717, 429, 800,  277, 1487, 1666, 1003, 1078,  901, 1717, 1209,  948, 1043,  834,],
 [ 915, 1092,  913, 1074,  829,  870,  809,  812,  680, 1077,  803,  675,  660,  709]])

print("Transformed by homography")
print(homography_trsf(pts, np.random.rand(3, 3)))
