
import numpy as np
import numpy.linalg as la
import pysal
import copy
import scipy.sparse as SP


def power_expansion(w, data, scalar, transpose=False, threshold=0.0000000001, max_iterations=None):
    """
    Compute the inverse of a matrix using the power expansion (Leontief
    expansion).  General form is:
    
        .. math:: 
            x &= (I - \rho W)^{-1}v = [I + \rho W + \rho^2 WW + \dots]v \\
              &= v + \rho Wv + \rho^2 WWv + \dots
 

    Parameters
    ----------

    w               : Pysal W object
                      nxn Pysal spatial weights object 

    data            : Numpy array
                      nx1 vector of data
    
    scalar          : float
                      Scalar value (typically rho or lambda)

    transpose       : boolean
                      If True then post-multiplies the data vector by the
                      inverse of the spatial filter, if false then
                      pre-multiplies.

    threshold       : float
                      Test value to stop the iterations. Test is against
                      sqrt(increment' * increment), where increment is a
                      vector representing the contribution from each
                      iteration.

    max_iterations  : integer
                      Maximum number of iterations for the expansion. 


    Examples
    --------

    >>> import numpy, pysal
    >>> import numpy.linalg as la
    >>> np.random.seed(10)
    >>> w = pysal.lat2W(5, 5)
    >>> w.transform = 'r'
    >>> data = np.random.randn(w.n)
    >>> data.shape = (w.n, 1)
    >>> rho = 0.4
    >>> inv_pow = power_expansion(w, data, rho)
    >>> # regular matrix inverse
    >>> matrix = np.eye(w.n) - (rho * w.full()[0])
    >>> matrix = la.inv(matrix)
    >>> inv_reg = np.dot(matrix, data)
    >>> np.allclose(inv_pow, inv_reg, atol=0.0001)
    True
    >>> inv_cg = inverse_cg(w, data, rho)
    >>> # test the transpose version
    >>> inv_pow = power_expansion(w, data, rho, transpose=True)
    >>> inv_reg = np.dot(data.T, matrix)
    >>> np.allclose(inv_pow, inv_reg, atol=0.0001)
    True


    """
    if transpose:
        lag = rev_lag_spatial
        data = data.T
    else:
        lag = pysal.lag_spatial
    running_total = copy.copy(data)
    count = 1
    test = 10000000
    if max_iterations == None:
        max_iterations = 1000000
    while test > threshold and count <= max_iterations:
        vector = (scalar**count) * data
        increment = lag(w, vector)
        running_total += increment
        test = la.norm(increment)
        count += 1
    return running_total


def rev_lag_spatial(w, y):
    """
    Helper function for power_expansion.  This reverses the usual lag operator
    (pysal.lag_spatial) to post-multiply a vector by a sparse W.

    Parameters
    ----------

    w : W
        weights object
    y : array
        variable to take the lag of (note: assumed that the order of y matches
        w.id_order)

    Returns
    -------

    yw : array
         array of numeric values

    Examples
    --------
    Tests for this function are in power_expansion()

    """
    return y * w.sparse


def inverse_scg(w, data, scalar, transpose=False, symmetric=False,\
               threshold=0.001,\
               max_iterations=None):
    
    multiplier = SP.identity(w.n) - (scalar*w.sparse)       # A      n x n   
    count = 0                                               # k      scalar (step 1)
    run_tot = copy.copy(data)                               # z_k    n x 1  (step 1)
    residuals = data - multiplier * run_tot                 # r_k    n x 1  (step 2)
    test1 = la.norm(residuals)                              # G_k    scalar (step 3)
    directions = copy.copy(residuals)                       # d_k    n x 1  (step 6)
    while test1 > threshold:                                #               (step 4)
        count += 1                                          #               (step 5)
        changes = multiplier * directions                   # t      n x 1  (step 7)
        intensity = test1 / (np.dot(directions.T, changes)) # alpha  scalar (step 8)
        int_dir = intensity * directions                    #               (step 8)
        run_tot += int_dir                                  #               (step 8)
        residuals -= int_dir                                #               (step 8)
        test2 = la.norm(residuals)                          #               (step 3)
        directions = residuals + ((test1/test2)*directions) #               (step 6)
        test1 = test2
        print test1
    return run_tot


def inverse_cg(w, data, scalar, transpose=False, symmetric=False,\
               threshold=1.0000000000000001e-05,\
               max_iterations=None):
    """
    Compute the inverse of a matrix using the conjugant gradient method.


    Parameters
    ----------

    w               : Pysal W object
                      nxn Pysal spatial weights object 

    data            : Numpy array
                      nx1 vector of data
    
    scalar          : float
                      Scalar value (typically rho or lambda)

    transpose       : boolean
                      If True then transposes the weights object.

    symmetric       : boolean
                      Boolean indicating if the weights matrix is symmetric. 
                      If True then function uses classic conjugant gradient 
                      method. If False then it uses the more general 
                      biconjugate gradient stabilized method. The general
                      method can be applied to a symmetric W, but it is
                      slower.

    threshold       : float
                      Test value to stop iterating.

    max_iterations  : integer
                      Maximum number of iterations.


    Examples
    --------

    >>> import numpy, pysal
    >>> import numpy.linalg as la
    >>> np.random.seed(10)
    >>> w = pysal.lat2W(5, 5)
    >>> w.transform = 'r'
    >>> data = np.random.randn(w.n)
    >>> data.shape = (w.n, 1)
    >>> rho = 0.4
    >>> inv_cg = inverse_cg(w, data, rho)
    >>> # regular matrix inverse
    >>> matrix = np.eye(w.n) - (rho * w.full()[0])
    >>> matrix = la.inv(matrix)
    >>> inv_reg = np.dot(matrix, data)
    >>> np.allclose(inv_cg, inv_reg, atol=0.0001)
    True
    >>> # test the transpose version
    >>> inv_pow = inverse_cg(w, data, rho, transpose=True)
    >>> inv_reg = np.dot(data.T, matrix)
    >>> np.allclose(inv_pow, inv_reg.T, atol=0.0001)
    True

    """
    identity = SP.identity(w.n)
    if transpose:
        multiplier = identity - (scalar*w.sparse.transpose())
    else:
        multiplier = identity - (scalar*w.sparse)
    if symmetric:
        cg_result =  SP.linalg.cg(multiplier, data, tol=threshold, maxiter=max_iterations)[0]
    else:
        cg_result =  SP.linalg.bicgstab(multiplier, data, tol=threshold, maxiter=max_iterations)[0]
    cg_result.shape = (w.n, 1)
    return cg_result

def regular_inverse(w, data, scalar, inv_type, transpose=False):
    """
    Temporary function for testing purposes. Will be deleted once the main
    functions work.
    """
    
    matrix = np.eye(w.n) - (scalar * w.full()[0])
    if inv_type == 'inv':
        matrix = la.inv(matrix)
    elif inv_type == 'pinv':
        matrix = la.pinv(matrix)
    if transpose:
        return np.dot(data.T, matrix)
    else:
        return np.dot(matrix, data)


if __name__ == '__main__':
    import doctest
    doctest.testmod()
    
    import time
    shape = 5
    w = pysal.lat2W(shape, shape)
    w.transform = 'r'
    d = np.random.randn(w.n)
    #d = np.random.randn(10)
    #d = np.append(d, [0]*15)
    #d = np.random.permutation(d)
    d.shape = (w.n, 1)
    time1 = time.time()
    #i_pe = power_expansion(w, d, 0.4)
    #i_cg = inverse_cg(w, d, 0.4)
    #i_cg = inverse_cg(w, d, 0.4, symmetric=True)
    t2 = time.time() - time1
    #print w.n, w.transform, t2
    



