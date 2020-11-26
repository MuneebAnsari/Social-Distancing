import cv2 
import numpy as np 

def homography_trsf(pts, M):
    '''
    Transforms the given points through homography 
    
    Params
    _________
    pts: 2xN matrix, where each column represents a point 
    M: 3x3 homography transformation 

    Returns
    _________
    pts: 2xN matrix representing transformed points
    '''
    N = pts.shape[1]
    homogeneous_coord = np.concatenate([pts, np.ones(1, N)], axis=0)
    world_coord = M @ homogeneous_coord
    world_coord = world_coord / world_coord[2, :]
    return world_coord[:2, :]

def find_close(pts, threshold=1, verbose=True):
    '''
    Find the set of index that are too close to other points. 

    Params
    _________
    pts: 2xN matrix, where each column represents a point 
    threshold: The threshold for closeness. 
    verbose: Whether computed distances are printed

    _________
    indices: The set of index too close to other points
    '''
    N = pts.shape[1]
    dists = np.zeros((N, N))
    np.fill_diagonal(dists, np.inf)
    indices = []

    # Compute Euclidean distances
    for i in range(N):
        for j in range(i+1, N):
            cur_diff = pts[:, i] - pts[:, j]
            cur_dist = np.sum(cur_diff * cur_diff) ** (1/2)
            dists[i, j] = cur_dist
    
    dists += dists.T 
    if verbose:
        print("Computed distances:\n{}".format(dists))
    
    # Collecting close indices 
    for i in range(N):
        if any(dists[i, :] < threshold):
            indices.append(i)
    
    return indices
    