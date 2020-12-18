import numpy as np 
from scipy import optimize
from homography_utils import get_homography, compute_homography, get_sap_homography

def cost(x, M, sap=False):
    '''
    Build homography M' based on parameters in x. 
    Return the L2 norm of (M - M') as cost 

    Params
    _________
    x: length of 9 vectors. f, px, py, angles, translations 
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

### Parameters
tol = 1e-10

### Testing method, random run 
M = get_homography(1, 0, 0, np.random.rand(3), np.random.rand(3))

### Testing whether M can be parametrized 
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

# TODO: Check SAP form is correct
# TODO: Code random initialization optimization