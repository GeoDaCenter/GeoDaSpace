import copy
import numpy as np
import pysal
import numpy.linalg as la
import twosls as TwoSLS
import ols as OLS

class TwoSLS_Spatial(TwoSLS.TwoSLS):
    """
    Spatial two stage least squares (2SLS). Also accommodates the case of
    endogenous explanatory variables.  Note: pure non-spatial 2SLS can be
    using the class Two_SLS
    
    Spatial 2SLS class for end-user (gives back only results and diagnostics)
    """
    def __init__(self):



        # Need to check the shape of user input arrays, and report back descriptive
        # errors when necessary

        # check the x array for a vector of ones... do we take the spatial lag
        # of these if W is not binary or row standardized?

        pass

class TwoSLS_Spatial_dev(TwoSLS.TwoSLS_dev):
    """
    Spatial 2SLS class to do all the computations

    NOTE: no consistency checks
    
    Maximal Complexity: 

    Parameters
    ----------

    x           : array
                  nxk array of independent variables, including endogenous
                  variables (assumed to be aligned with y)
    y           : array
                  nx1 array of dependent variable
    w           : spatial weights object
                  pysal spatial weights object
    h           : array
                  optional nxl array of instruments; typically this includes 
                  all exogenous variables from x and instruments; if set to
                  None then only spatial lagged 
    w_lags      : integer
                  Number of spatial lags of the exogenous variables. Kelejian
                  et al. (2004) recommends w_lags=2, which is the default.
    constant    : boolean
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
    >>> w = pysal.rook_from_shapefile("examples/columbus.shp")
    >>> w.transform = 'r'
    >>> reg_justSpatial=TwoSLS_Spatial_dev(X, y, w, w_lags=2)
    >>> reg_justSpatial.betas
    array([[ 45.45909249],
           [  0.41929355],
           [ -1.0410089 ],
           [ -0.25953844]])
    >>> reg_endogenous=TwoSLS_Spatial_dev(X, y, w, h, w_lags=2)
    """

    def __init__(self, x, y, w, h=None, w_lags=2, constant=True):
        if type(h).__name__ == 'ndarry': # spatial and non-spatial instruments
            h = self.get_lags(h, w, w_lags)
        elif h == None:                  # spatial instruments only
            h = self.get_lags(x, w, w_lags)
        yl = pysal.lag_spatial(w, y)
        x = np.hstack((yl, x))
        TwoSLS.TwoSLS_dev.__init__(self, x, y, h, constant)
       
    def get_lags(self, rhs, w, w_lags):
        rhsl = copy.copy(rhs)
        for i in range(w_lags):
            rhsl = pysal.lag_spatial(w, rhsl)
            rhs = np.hstack((rhs, rhsl))    
        return rhs

    @property
    def vm(self):
        # follows stsls in R spdep
        if 'vm' not in self._cache:
            self._cache['vm'] = np.dot(self.sig2n_k, self.xptxpi)
        return self._cache['vm']



def _test():
    import doctest
    doctest.testmod()

if __name__ == '__main__':
    _test()



