import copy
import numpy as np
import pysal
import numpy.linalg as la
import twosls as TSLS
import robust as ROBUST

class STSLS(TSLS.TSLS):
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

        # check capitalization in the string passed to robust parameter. 

        pass

class BaseSTSLS(TSLS.BaseTSLS):
    """
    Spatial 2SLS class to do all the computations

    NOTE: no consistency checks
    
    Maximal Complexity: 

    Parameters
    ----------

    y           : array
                  nx1 array of dependent variable
    x           : array
                  array of independent variables, excluding endogenous
                  variables (assumed to be aligned with y)
    w           : spatial weights object
                  pysal spatial weights object
    yend        : array
                  endogenous variables (assumed to be aligned with y)
    q           : array
                  array of instruments (assumed to be aligned with yend); 
    w_lags      : integer
                  Number of spatial lags of the exogenous variables. 
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
    z           : array
                  n*k array of independent variables, including endogenous
                  variables (assumed to be aligned with y)                  
    h           : array
                  nxl array of instruments, this includes all exogenous variables 
                  from x and instruments
    betas       : array
                  kx1 array with estimated coefficients
    u           : array
                  nx1 array of residuals (based on original x matrix)
    predy       : array
                  nx1 array of predicted values (based on original x matrix)
    n           : int
                  Number of observations
    k           : int
                  Number of variables, including exogenous and endogenous variables
    utu         : float
                  Sum of the squared residuals
    sig2        : float
                  Sigma squared with n in the denominator
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
    >>> import diagnostics as D
    >>> w = pysal.rook_from_shapefile("examples/columbus.shp")
    >>> w.transform = 'r'
    >>> db=pysal.open("examples/columbus.dbf","r")
    >>> y = np.array(db.by_col("CRIME"))
    >>> y = np.reshape(y, (49,1))
    >>> # no non-spatial endogenous variables
    >>> X = []
    >>> X.append(db.by_col("INC"))
    >>> X.append(db.by_col("HOVAL"))
    >>> X = np.array(X).T
    >>> reg=BaseSTSLS(y, X, w, w_lags=2)
    >>> reg.betas
    array([[ 45.45909249],
           [ -1.0410089 ],
           [ -0.25953844],
           [  0.41929355]])
    >>> D.se_betas(reg)
    array([ 11.19151175,   0.38861224,   0.09240593,   0.18758518])
    >>> reg=BaseSTSLS(y, X, w, w_lags=2, robust='white')
    >>> reg.betas
    array([[ 45.45909249],
           [ -1.0410089 ],
           [ -0.25953844],
           [  0.41929355]])
    >>> D.se_betas(reg)
    array([ 10.93497906,   0.49943339,   0.17217193,   0.19588229])
    >>> reg=BaseSTSLS(y, X, w, w_lags=2, robust='gls')
    >>> reg.betas
    array([[ 51.16882977],
           [ -1.12721019],
           [ -0.28543096],
           [  0.32904005]])
    >>> D.se_betas(reg)
    array([ 7.2237932 ,  0.4297434 ,  0.15201063,  0.13208009])
    >>> # instrument for HOVAL with DISCBD
    >>> X = np.array(db.by_col("INC"))
    >>> X = np.reshape(X, (49,1))
    >>> yd = np.array(db.by_col("HOVAL"))
    >>> yd = np.reshape(yd, (49,1))
    >>> q = np.array(db.by_col("DISCBD"))
    >>> q = np.reshape(q, (49,1))
    >>> reg=BaseSTSLS(y, X, w, yd, q, w_lags=2)

    References
    ----------

    .. [1] Kelejian, H.H., Prucha, I.R. and Yuzefovich, Y. (2004)
    "Instrumental variable estimation of a spatial autorgressive model with
    autoregressive disturbances: large and small sample results". Advances in
    Econometrics, 18, 163-198.
    """

    def __init__(self, y, x, w, yend=None, q=None, w_lags=1, constant=True, robust=None):
        yl = pysal.lag_spatial(w, y)
        if issubclass(type(yend), np.ndarray):  # spatial and non-spatial instruments
            lag_vars = np.hstack((x, q))
            spatial_inst = self.get_lags(lag_vars, w, w_lags)
            q = np.hstack((q, spatial_inst))
            yend = np.hstack((yend, yl))
        elif yend == None:                   # spatial instruments only
            q = self.get_lags(x, w, w_lags)
            yend = yl
        else:
            raise Exception, "invalid value passed to yend"
        TSLS.BaseTSLS.__init__(self, y, x, yend, q, constant, robust)
        self.sig2 = self.sig2n_k
        if robust == 'gls':
            self.vm = self.vm_gls
        elif robust == 'white':
            self.vm = self.vm_white
        
    def get_lags(self, x, w, w_lags):
        lag = pysal.lag_spatial(w, x)
        spat_inst = lag
        for i in range(w_lags-1):
            lag = pysal.lag_spatial(w, lag)
            spat_inst = np.hstack((spat_inst, lag))
        return spat_inst

    @property
    def vm_gls(self):
        # follows stsls in R spdep
        if 'vm' not in self._cache:
            self._cache['vm'] = self.xptxpi
        return self._cache['vm']

    @property
    def vm_white(self):
        # follows stsls in R spdep
        if 'vm' not in self._cache:
            v = ROBUST.get_omega(self.z, self.u)
            xptxpiv = np.dot(self.xptxpi, v)
            self._cache['vm'] = np.dot(xptxpiv, self.xptxpi)
        return self._cache['vm']


    #### The results currently match stsls in R spdep for both forms of
    #### robustness (gls and white) for the case of no additional endogenous 
    #### variables.  I have not checked any of the results when non-spatial
    #### instruments are included.


def _test():
    import doctest
    doctest.testmod()

if __name__ == '__main__':
    _test()



