import numpy as np
import numpy.linalg as la


class OLS:
    """
    OLS class for end-user (gives back only results ans diagnostics)
    """
    def __init__(self):
        pass

class OLS_dev:
    """
    OLS class to do all the computations

    NOTE: no consistency checks
    
    Maximal Complexity: O(n^3)

    Parameters
    ----------

    x       : array
              nxk array of independent variables (assumed to be aligned with y)
    y       : array
              nx1 array of dependent variable
    constant: boolean
              If true it appends a vector of ones to the independent variables
              to estimate intercept (set to True by default)

    Attributes
    ----------

    x       : array
              nxk array of independent variables (assumed to be aligned with y)
    y       : array
              nx1 array of dependent variable
    betas   : array
              kx1 array with estimated coefficients
    xt      : array
              kxn array of transposed independent variables
    xtx     : array
              kxk array
    xtxi    : array
              kxk array of inverted xtx
    u       : array
              nx1 array of residuals
    predy   : array
              nx1 array of predicted values
    n       : int
              Number of observations
    k       : int
              Number of variables
    utu     : float
              Sum of the squared residuals
    sig2    : float
              Sigma squared with n in the denominator
    sig2n_k : float
              Sigma squared with n-k in the denominator

              .. math::
                
                    \sigma^2 = \dfrac{\tilde{u}' \tilde{u}}{N}

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
    >>> ols=OLS_dev(X,y)
    >>> ols.betas
    array([[ 68.6189611 ],
           [ -1.59731083],
           [ -0.27393148]])
    
    """
    def __init__(self,x,y,constant=True):
        if constant:
            x = np.hstack((np.ones(y.shape),x))
        xt = np.transpose(x)
        xtx = np.dot(xt,x)
        xtxi = la.inv(xtx)
        xty = np.dot(xt,y)
        self.betas = np.dot(xtxi,xty)
        self.xtxi = xtxi
        self.xtx = xtx
        self.xt = xt
        predy = np.dot(x,self.betas)
        u = y-predy
        self.u = u
        self.predy = predy
        self.y = y
        self.x = x
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
    @property
    def vm(self):
        if 'vm' not in self._cache:
            estSig2= self.utu / (self.n-self.k)
            self._cache['vm'] = np.dot(estSig2, self.xtxi)
        return self._cache['vm']
    

def _test():
    import doctest
    doctest.testmod()

if __name__ == '__main__':
    _test()



