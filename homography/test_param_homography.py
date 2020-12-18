import numpy as np 
from scipy import optimize
from homography_utils import get_homography, compute_homography, get_sap_homography

def cost(x, M, sap=False):
    '''
    Build homography M' based on parameters in x. 
    Return the L2 norm of (M - M') as cost 

    Params
    _________
    x: parameters of the homography matrix. f, px, py, angles, translations 
        This is currently overparameterized because H has only 8 degree of freedom. 
        But it's too complex to figure the angles that's overparametrized. 
    M: The homography matrix 
    '''
    # Obtain the difference

    if sap:
        Mp = get_sap_homography(x[0], x[1], x[2:4], x[4], x[5], x[6:])
    else: 
        Mp = get_homography(x[0], x[1], x[2], x[3:6], x[6:]) 
    Mp = Mp / np.sum(Mp * Mp)**(1/2) # Choose norm of Mp
    diff = M - Mp
    
    # Compute cost 
    l2 = np.sum(diff * diff) ** (1/2)
    return l2 

def test_parametrization(M, sap, attempts=100, tol=1e-6, suc_tol=1e-4):
    '''
    Do multiple attempts to parametrize M with two parametrization

    Params
    _________
    M: The homography matrix to be parametrized
    sap: Whether to use the SAP decomposition or the 3x4 -> 3x3 parametrization from lecture 
    attempts: number of tries

    Returns
    _________
    rate: The success rate of the parametrization 
    ''' 
    success = 0
    for i in range(attempts):
        if sap:
            x0 = np.random.rand(8) 
            fun = lambda x: cost(x, M, sap=True)
        else:
            x0 = np.random.rand(9) 
            fun = lambda x: cost(x, M, sap=False)
        sol = optimize.minimize(fun, x0, tol=tol)
        if sol.fun < suc_tol:
            success += 1
    return success / attempts

### Parameters
tol = 1e-12
attempts = 10 

### Testing method, random run 
M = get_homography(1, 0, 0, np.random.rand(3), np.random.rand(3))

### Testing whether M can be parametrized 
# Here I used the actual points picked from an image
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
M = compute_homography(src_pts.T, dst_pts.T)

##### Minimization 
### Lecture form of M
x0 = np.random.rand(9) 
fun = lambda x: cost(x, M, sap=False)
sol = optimize.minimize(fun, x0, tol=tol)

print("Converged Solution. Lecture form: {}".format(sol.x))
print("Converged L2 norm. Lecture form: {}".format(sol.fun))

### SAP form of M
x0 = np.random.rand(8) 
fun = lambda x: cost(x, M, sap=True)
sol = optimize.minimize(fun, x0, tol=tol)
print("Converged Solution. SAP form: {}".format(sol.x))
print("Converged L2 norm. SAP form: {}".format(sol.fun))

print("Percentage of success parametrization using lecture form: {}".format(test_parametrization(M, sap=False, tol=tol, attempts=attempts)))
print("Percentage of success parametrization using lecture form: {}".format(test_parametrization(M, sap=True, tol=tol, attempts=attempts)))