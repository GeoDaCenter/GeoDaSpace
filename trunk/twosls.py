import numpy as np
import numpy.linalg as la
import ols as OLS

class TwoSLS:
    """
    2SLS class for end-user (gives back only results and diagnostics)
    """
    def __init__(self):

        # Need to check the shape of user input arrays, and report back descriptive
        # errors when necessary
        pass

class TwoSLS_dev(OLS.OLS_dev):
    """
    2SLS class to do all the computations

    NOTE: no consistency checks
    
    Maximal Complexity: 

    Parameters
    ----------

    x           : array
                  nxk array of independent variables, including endogenous
                  variables (assumed to be aligned with y)
    y           : array
                  nx1 array of dependent variable
    h           : array
                  nxl array of instruments
    constants   : boolean
                  If true it appends a vector of ones to the independent variables
                  to estimate intercept (set to True by default)

    Attributes
    ----------

    x           : array
                  nxk array of independent variables (assumed to be aligned with y)
    y           : array
                  nx1 array of dependent variable
    h           : array
                  nxl array of instruments
    betas       : array
                  kx1 array with estimated coefficients
    xt          : array
                  kxn array of transposed independent variables
    xtx         : array
                  kxk array
    xtxi        : array
                  kxk array of inverted xtx
    u           : array
                  nx1 array of residuals (based on original x matrix)
    u_ols       : array
                  nx1 array of residuals (based on transformed x matrix)
    predy_ols   : array
                  nx1 array of predicted values (based on transformed x matrix)
    predy       : array
                  nx1 array of predicted values (based on original x matrix)
    n           : int
                  Number of observations
    k           : int
                  Number of variables
    utu         : float
                  Sum of the squared residuals
    sig2        : float
                  Sigma squared with n in the denominator
    sig2n_k     : float
                  Sigma squared with n-k in the denominator

              .. math::
                
                    \sigma^2 = \dfrac{\tilde{u}' \tilde{u}}{N}
    m       : array
              Matrix M

              .. math::

                    M = I - X(X'X)^{-1}X'


    Examples
    --------

    >>> import numpy as np
    >>> import pysal
    >>> db=pysal.open("examples/columbus.dbf","r")
    >>> y = np.array(db.by_col("CRIME"))
    >>> y = np.reshape(y, (49,1))
    >>> X = []
    >>> X.append(db.by_col("INC"))
    >>> X.append(db.by_col("HOVAL"))
    >>> X = np.array(X).T
    >>> h = db.by_col("DISCBD")
    >>> h = db.by_col("PERIMETER")
    >>> h = np.array(h)
    >>> h = np.reshape(h, (49,1))
    >>> reg=TwoSLS_dev(X,y,h)
    >>> reg.betas
    array([[ 68.6189611 ],
           [ -1.59731083],
           [ -0.27393148]])
    
    """
    def __init__(self, x, y, h, constant=True):
        if constant:
            x = np.hstack((np.ones(y.shape),x))
        # x_hat = Z(Z'Z)^-1 Z'X
        z = np.hstack((x, h))
        print z
        print '***********************************************'
        zpz = np.dot(z.T,z)   # k+l x k+l 
        print zpz
        print '***********************************************'
        zpzi = la.inv(zpz)   # k+l x k+l 
        print zpzi
        print '***********************************************'
        zpx = np.dot(z.T, x)           # k+l x k
        print zpx
        print '***********************************************'
        zpzi_zpx = np.dot(zpzi, zpx)   # k+l x k
        print zpzi_zpx
        print '***********************************************'
        x_hat = np.dot(z, zpzi_zpx)    #   n x k

        OLS.OLS_dev.__init__(self, x_hat, y, constant=False)
        self.predy_ols = self.predy         # using transformed data
        self.u_ols = self.u                 # using transformed data
        self.predy = np.dot(x,self.betas)   # using original data
        self.u = y - self.predy               # using original data
        self.x = x
        self.h = h


def _test():
    import doctest
    doctest.testmod()

if __name__ == '__main__':
    _test()



