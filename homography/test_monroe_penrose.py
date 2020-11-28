import numpy as np
from homography_utils import *

### Preparing points 
src = np.array([[1], [3]])
dst = np.array([[5], [7]])

print(compute_homography_build_A(src, dst))

src = np.random.rand(2, 4)
dst = np.random.rand(2, 4)
print(compute_homography(src, dst))
