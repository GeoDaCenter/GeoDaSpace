import numpy as np
import numpy.linalg as la
import ols as OLS
import robust as ROBUST
import user_output as USER


class TSLS_dev(OLS.OLS_dev, OLS.Regression_Props):
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
                  nxl array of instruments; typically this includes all
                  exogenous variables from x and instruments
    constant    : boolean
                  If true it appends a vector of ones to the independent variables
                  to estimate intercept (set to True by default)

    robust      : string
                  If 'white' then a White consistent estimator of the
                  variance-covariance matrix is given. If 'gls' then
                  generalized least squares is performed resulting in new
                  coefficient estimates along with a new variance-covariance
                  matrix. 

    Attributes
    ----------

    x           : array
                  nxk array of independent variables (assumed to be aligned with y)
    y           : array
                  nx1 array of dependent variable
    h           : array
                  nxl array of instruments
    z           : array
                  array of x and instruments appended
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
    vm          : array
                  Variance-covariance matrix (kxk)
    mean_y      : float
                  Mean of the dependent variable
    std_y       : float
                  Standard deviation of the dependent variable


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
    >>> # instrument for HOVAL with DISCBD
    >>> h = []
    >>> h.append(db.by_col("INC"))
    >>> h.append(db.by_col("DISCBD"))
    >>> h = np.array(h).T
    >>> reg = TSLS_dev(X, y, h)
    >>> reg.betas
    array([[ 88.46579584],
           [  0.5200379 ],
           [ -1.58216593]])
    
    """
    def __init__(self, x, y, h, constant=True, robust=None):
        self.h = h
        if constant:
            x = np.hstack((np.ones(y.shape),x))
            z = np.hstack((np.ones(y.shape),h))
            # h = np.hstack((np.ones(y.shape),h)) # jy
        else:
            z = h
        self.z = z
        ztz = np.dot(z.T,z)
        ztzi = la.inv(ztz)
        ztx = np.dot(z.T, x)
        ztzi_ztx = np.dot(ztzi, ztx)
        x_hat = np.dot(z, ztzi_ztx)          # x_hat = Z(Z'Z)^-1 Z'X
        OLS.OLS_dev.__init__(self, x_hat, y, constant=False)
        self.xptxpi = self.xtxi              # using predicted x (xp)
        self.u_2nd_stage = self.u
        self.set_x(x)  # reset x, xtx and xtxi attributes to use original data
        self.predy = np.dot(x, self.betas)   # using original data
        self.u = y - self.predy              # using original data
        if robust == 'gls':
            self.betas, self.xptxpi = ROBUST.gls_dev(x, y, z, self.u)
            self.predy = np.dot(x, self.betas)   # using original data and GLS betas
            self.u = y - self.predy              # using original data and GLS betas
            ### need to verify the VM for the non-spatial case
        OLS.Regression_Props()


        #### GLS and White robust 2SLS was implemented for the spatial case,
        #### and results match there.  I have not tested the robust results
        #### for the non-spatial case.
        '''
        ## 2SLS in one step: jy
        z = x
        zt = z.T
        ht = h.T
        hth = np.dot(ht,h)
        hthi = la.inv(hth)
        zth = np.dot(zt,h)
        ztht = zth.T
        
        factor_1 = np.dot(zth,hthi)
        factor_2 = np.dot(factor_1,ht)
        factor_2 = np.dot(factor_2,z)
        factor_2 = la.inv(factor_2)        
        factor_2 = np.dot(factor_2,factor_1)        
        factor_2 = np.dot(factor_2,ht)       
        rs = np.dot(factor_2,y)
        '''
        
    @property
    def vm(self):
        if 'vm' not in self._cache:
            self._cache['vm'] = np.dot(self.sig2, self.xptxpi)
        return self._cache['vm']
    
    
class TSLS(TSLS_dev, USER.Diagnostic_Builder):
    """
    2SLS class for end-user (gives back only results and diagnostics)

        # Need to check the shape of user input arrays, and report back descriptive
        # errors when necessary

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
    >>> # instrument for HOVAL with DISCBD
    >>> h = []
    >>> h.append(db.by_col("INC"))
    >>> h.append(db.by_col("DISCBD"))
    >>> h = np.array(h).T
    >>> reg = TSLS(X, y, h, name_x=['INC','HOVAL'], name_y='CRIME', name_h=['INC','DISCBD'], name_ds='columbus')
    >>> reg.betas
    array([[ 88.46579584],
           [  0.5200379 ],
           [ -1.58216593]])
    
    """
    def __init__(self, x, y, h, constant=True, robust=None, name_x=None,\
                        name_y=None, name_h=None, name_ds=None, vm=False,\
                        pred=False):
        TSLS_dev.__init__(self, x, y, h, constant)
        self.title = "TWO STAGE LEAST SQUARES"
        self.name_ds = name_ds
        self.name_y = name_y
        self.name_x = name_x
        self.name_h = name_h
        USER.Diagnostic_Builder.__init__(self, constant=constant, vm=vm,\
                                            pred=pred, instruments=True)



def _test():
    import doctest
    doctest.testmod()

if __name__ == '__main__':
    _test()



