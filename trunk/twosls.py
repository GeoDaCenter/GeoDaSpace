import numpy as np
import numpy.linalg as la
import ols as OLS

class TwoSLS:
    """
    2SLS class for end-user (gives back only results and diagnostics)
    """
    def __init__(self):
        pass

class TwoSLS_dev:
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
    >>> h = db.by_col()
    >>> h = np.array(h)
    >>> reg=TwoSLS_dev(X,y)
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
        zpzi = la.inv(ntp.inner(z,z))  # k+l x k+l 
        zpx = np.dot(z.T, x)           # k+l x k
        zpzi_zpx = np.dot(zpzi, zpx)   # k+l x k
        x_hat = np.dot(z, zpzi_zpx)    #   n x k

        ols = OLS.ols_dev(x_hat, y, constant=False)
        self.betas = ols.betas
        self.predy = np.dot(x,self.betas)  # using original data
        self.u = y-predy                   # using original data
        self.predy_ols = ols.predy         # using transformed data
        self.u_ols = ols.u                 # using transformed data
        self.y = y
        self.x = x
        self.h = h
        self.n, self.k = x.shape
        self._cache = {}

    @property
    def utu(self):
        if 'utu' not in self._cache:
            self._cache['utu'] = np.sum(self.u**2)
        return self._cache['utu']

    @property
    def sig2(self):
        if 'sig2' not in self._cache:
            self._cache['sig2'] = self.utu / self.n
        return self._cache['sig2']

    @property
    def m(self):
        if 'm' not in self._cache:
            xtxixt = np.dot(self.xtxi,self.xt)
            xxtxixt = np.dot(self.x, xtxixt)
            self._cache['m'] = np.eye(self.n) - xxtxixt
        return self._cache['m']

    # check this
    @property
    def vm(self):
        if 'vm' not in self._cache:
            estSig2= self.utu / (self.n-self.k)
            self._cache['vm'] = np.dot(estSig2, self.xtxi)
        return self._cache['vm']
    
    @property
    def mean_y(self):
        if 'mean_y' not in self._cache:
            self._cache['mean_y']=np.mean(self.y)
        return self._cache['mean_y']
    
    @property
    def std_y(self):
        if 'std_y' not in self._cache:
            self._cache['std_y']=np.std(self.y)
        return self._cache['std_y']
    

def _test():
    import doctest
    doctest.testmod()

if __name__ == '__main__':
    _test()



