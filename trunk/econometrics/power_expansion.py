
import numpy as np
import numpy.linalg as la
import pysal
import copy
import scipy.sparse as SP


def power_expansion(w, data, scalar, transpose=False, threshold=0.0000001, max_iterations=100):
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
                      Maximum number of iterations for the expansion


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


def regular_inverse(w, data, scalar, transpose=False):
    """
    Temporary function for testing purposes. Will be deleted once the main
    functions work.
    """
    
    matrix = np.eye(w.n) - (scalar * w.full()[0])
    matrix = la.pinv(matrix)
    if transpose:
        return np.dot(data.T, matrix)
    else:
        return np.dot(matrix, data)


if __name__ == '__main__':
    import doctest
    doctest.testmod()



