import cv2 
import numpy as np 

def compute_homography_build_A(src, dst):
    '''
    Compute the matrix A for formulation Ah = 0. 

    Params
    _________
    src/dst: 2xN matrix, where each column represents a point

    Returns
    _________
    A: 2N x 9 matrix 
    '''
    def prepare_pt(spt, dpt):
        '''
        Prepare 
        [[x_i, y_i, 1, 0, 0, 0, -x_i' x_i, -x_i' y_i, -x_i],
        [0, 0, 0, x_i, y_i, 1, -y_i' x_i, -y_i' y_i, -y_i]]

        Params
        _________
        spt/dpt: 2, vectors representing points
        '''
        # Add 1 and reshape to row vector
        spt_1 = np.reshape(np.concatenate([spt, [1]], axis=0), (1, 3)) 

        spt_up = np.concatenate([spt_1, np.zeros((1, 3))], axis=0)
        spt_dn = np.concatenate([np.zeros((1, 3)), spt_1], axis=0)
        prod_mat = np.einsum('i,j->ij', -dpt, spt)
        dpt_neg = - np.reshape(dpt, (2, 1))

        re = np.concatenate([
            spt_up, spt_dn, prod_mat, dpt_neg
        ], axis=1)
        return re 

    N = src.shape[1]

    A = np.zeros((2*N, 9))
    for i in range(N):
        A[2*i:2*i+2, :] = prepare_pt(src[:, i], dst[:, i])
    
    return A

def compute_homography(src, dst):
    '''
    Solving for the homography matrix needed to take src points to dst points
    Use the formulation Ah = 0
    where matrix A takes the form 
    [
        ...
        [x_i, y_i, 1, 0, 0, 0, -x_i' x_i, -x_i' y_i, -x_i],
        [0, 0, 0, x_i, y_i, 1, -y_i' x_i, -y_i' y_i, -y_i],
        ...
    ] 
    and h contains the coefficients of homography matirx

    Params
    _________
    src/dst: 2xN matrix, where each column represents a point. N must be greater than 4. 

    Returns
    _________
    pts: 3x3 matrix representing homography transformation
    '''
    N = src.shape[1]
    if N < 4:
        raise ValueError("Homography matrix can only be computed given 4 data points")
    A = compute_homography_build_A(src, dst)
    M = A.T @ A
    eigvals, eigvecs = np.linalg.eigh(M)

    # Returning the eigenvector with smallest eigenvalue
    return np.reshape(eigvecs[:, 0], (3, 3))

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

def get_rotational_matrix(angs):
    '''
    Obtain the 3x3 3D rotational matrix given by euler angles angs. 
    Equation taken from https://en.wikipedia.org/wiki/Rotation_formalisms_in_three_dimensions
    
    Params
    _________
    angs: The three euler angles. 
    '''
    # Obtain three rotations 
    ax, ay, az = angs[0], angs[1], angs[2]
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(ax), -np.sin(ax)],
        [0, np.sin(ax), np.cos(ax)] 
    ])
    Ry = np.array([
        [np.cos(ay), 0, np.sin(ay)],
        [0, 1, 0],
        [-np.sin(ay), 0, np.cos(ay)] 
    ])
    Rz = np.array([
        [np.cos(az), -np.sin(az), 0],
        [-np.sin(az), np.cos(az), 0],
        [0, 0, 1] 
    ])
    # Do rotation in order
    return Rz @ Ry @ Rx 

def get_homography(f, px, py, angs, ts):
    '''
    Obtain the homography specified by these parameters 
    
    Params
    _________
    f, px, py: scalars. 
        Lens ratio & displacements for x or y
    angs/ts: vectors with three elements. 
        The euler angles and translation vector of camera 
    '''

    # The intrinsic matrix 
    K = np.array([
        [f, 0, px],
        [0, f, py],
        [0, 0, 1], 
    ])

    # The rotation matrix.
    R = get_rotational_matrix(angs)
    R = R[:, :2]

    # The displacement vector
    ts = np.reshape(ts, (3, 1))

    # The extrinsic matrix
    E = np.concatenate([R, ts], axis=1)
    return K @ E

def get_sap_homography(s, r_ang, t, k_diag, k_off, v):
    '''
    Return the homography transformation as the product of HS HA HP matrix 
    specified below 
    HS: [   [sR(r_ang), t12]
            [0^T 1]   ]
    HA: [   [K 0]
            [0 1] ]
    HP: [   [I 0]
            [v12 1] ]
    '''

    HS = np.array([
        [s*np.cos(r_ang), -s*np.sin(r_ang), t[0]],
        [s*np.sin(r_ang), s*np.cos(r_ang), t[1]],
        [0, 0, 1]
    ]) # 4 dof 
    HA = np.array([
        [k_diag, k_off, 0],
        [0, k_diag**(-1), 0],
        [0, 0, 1]
    ]) # 2 dof
    HP = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [v[0], v[1], 1]
    ])

    return HS @ HA @ HP
