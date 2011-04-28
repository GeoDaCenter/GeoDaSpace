import copy
import numpy as np
import pysal
import numpy.linalg as la
import twosls as TSLS
import robust as ROBUST
import user_output as USER
from utils import get_lags

class BaseGM_Lag(TSLS.BaseTSLS):
    """
    Spatial 2SLS class to do all the computations


    Parameters
    ----------

    y           : array
                  nx1 array of dependent variable
    x           : array
                  array of independent variables, excluding endogenous
                  variables
    w           : spatial weights object
                  pysal spatial weights object
    yend        : array
                  non-spatial endogenous variables [optional]
    q           : array
                  array of instruments for yend (note: this should not contain
                  any variables from x; spatial instruments are computed by 
                  default) [only if 'yend' passed]
    w_lags      : integer
                  Number of spatial lags of the exogenous variables to be
                  included as spatial instruments (default set to 1)
    constant    : boolean
                  If true it appends a vector of ones to the independent variables
                  to estimate intercept (set to True by default)
    robust      : string
                  If 'white' or 'hac' then a White consistent or HAC estimator
                  of the variance-covariance matrix is given. If 'gls' then
                  generalized least squares is performed resulting in new
                  coefficient estimates along with a new variance-covariance
                  matrix. 
    spat_lags   : string
                  If 'xq' (default) then spatial lags of all exogenous
                  variables are used as instruments (i.e. lags of x and q); if
                  'x' then just spatial lags of x are included
    wk          : spatial weights object
                  pysal kernel weights object

    Attributes
    ----------

    y           : array
                  nx1 array of dependent variable
    x           : array
                  array of independent variables (with constant added if
                  constant parameter set to True)
    z           : array
                  nxk array of variables (combination of x, yend and spatial
                  lag of y)
    h           : array
                  nxl array of instruments (combination of x, q, spatial lags)
    yend        : array
                  endogenous variables (including spatial lag)
    q           : array
                  array of external exogenous variables (including spatial
                  lags)
    betas       : array
                  kx1 array of estimated coefficients
    u           : array
                  nx1 array of residuals
    predy       : array
                  nx1 array of predicted values
    n           : int
                  Number of observations
    k           : int
                  Number of variables, including exogenous and endogenous
                  variables and constant
    kstar       : int
                  Number of endogenous variables. 
    zth         : array
                  z.T * h
    hth         : array
                  h.T * h
    htz         : array
                  h.T * z
    hthi        : array
                  inverse of h.T * h
    xp          : array
                  h * np.dot(hthi, htz)           
    xptxpi      : array
                  inverse of np.dot(xp.T,xp), used to compute vm
    pfora1a2    : array
                  used to compute a1, a2


    Examples
    --------

    >>> import numpy as np
    >>> import pysal
    >>> import pysal.spreg.diagnostics as D
    >>> w = pysal.rook_from_shapefile("examples/columbus.shp")
    >>> w.transform = 'r'
    >>> db=pysal.open("examples/columbus.dbf","r")
    >>> y = np.array(db.by_col("HOVAL"))
    >>> y = np.reshape(y, (49,1))
    >>> # no non-spatial endogenous variables
    >>> X = []
    >>> X.append(db.by_col("INC"))
    >>> X.append(db.by_col("CRIME"))
    >>> X = np.array(X).T
    >>> reg=BaseGM_Lag(y, X, w, w_lags=2)
    >>> reg.betas
    array([[  4.53017056e+01],
           [  6.20888617e-01],
           [ -4.80723451e-01],
           [  2.83622122e-02]])
    >>> D.se_betas(reg)
    array([ 18.69196608,   0.54769142,   0.19021045,   0.33120731])
    >>> reg=BaseGM_Lag(y, X, w, w_lags=2, robust='white')
    >>> reg.betas
    array([[  4.53017056e+01],
           [  6.20888617e-01],
           [ -4.80723451e-01],
           [  2.83622122e-02]])
    >>> D.se_betas(reg)
    array([ 26.46540238,   0.54128829,   0.20836574,   0.607184  ])
    >>> # instrument for HOVAL with DISCBD
    >>> X = np.array(db.by_col("INC"))
    >>> X = np.reshape(X, (49,1))
    >>> yd = np.array(db.by_col("CRIME"))
    >>> yd = np.reshape(yd, (49,1))
    >>> q = np.array(db.by_col("DISCBD"))
    >>> q = np.reshape(q, (49,1))
    >>> reg=BaseGM_Lag(y, X, w, yd, q, w_lags=2)

    References
    ----------

    .. [1] Kelejian, H.H., Prucha, I.R. and Yuzefovich, Y. (2004)
    "Instrumental variable estimation of a spatial autoregressive model with
    autoregressive disturbances: large and small sample results". Advances in
    Econometrics, 18, 163-198.
    """

    def __init__(self, y, x, w, yend=None, q=None, w_lags=1,\
                    constant=True, robust=None, spat_lags='xq', wk=None):
        yl = pysal.lag_spatial(w, y)
        if issubclass(type(yend), np.ndarray):  # spatial and non-spatial instruments
            if spat_lags == 'xq':
                lag_vars = np.hstack((x, q))
            elif spat_lags == 'x':
                lag_vars = np.hstack((x))
            else:
                raise Exception, "invalid value passed to spat_lags"
            spatial_inst = get_lags(w, lag_vars, w_lags)
            q = np.hstack((q, spatial_inst))
            yend = np.hstack((yend, yl))
        elif yend == None:                   # spatial instruments only
            q = get_lags(w, x, w_lags)
            yend = yl
        else:
            raise Exception, "invalid value passed to yend"
        TSLS.BaseTSLS.__init__(self, y, x, yend, q=q, constant=constant)
        self.sig2 = self.sig2n_k
        if robust == 'white':
            self.vm = self.vm_white       
            #self.vm = ROBUST.robust_vm(self, wk=wk)
        elif robust == 'hac':
            self.vm = ROBUST.robust_vm(self, wk=wk)
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


class GM_Lag(BaseGM_Lag, USER.DiagnosticBuilder):
    """
    Spatial two stage least squares (S2SLS). Also accommodates the case of
    endogenous explanatory variables.  Note: pure non-spatial 2SLS can be run
    using the class TSLS.
    


        # Need to check the shape of user input arrays, and report back descriptive
        # errors when necessary

        # check the x array for a vector of ones... do we take the spatial lag
        # of these if W is not binary or row standardized?

        # check capitalization in the string passed to robust parameter. 


    Examples
    --------

    >>> import numpy as np
    >>> import pysal
    >>> import pysal.spreg.diagnostics as D
    >>> w = pysal.rook_from_shapefile("examples/columbus.shp")
    >>> w.transform = 'r'
    >>> db=pysal.open("examples/columbus.dbf","r")
    >>> y = np.array(db.by_col("HOVAL"))
    >>> y = np.reshape(y, (49,1))
    >>> # no non-spatial endogenous variables
    >>> X = []
    >>> X.append(db.by_col("INC"))
    >>> X.append(db.by_col("CRIME"))
    >>> X = np.array(X).T
    >>> reg=GM_Lag(y, X, w, w_lags=2, name_x=['inc', 'crime'], name_y='hoval', name_ds='columbus')
    >>> reg.betas
    array([[  4.53017056e+01],
           [  6.20888617e-01],
           [ -4.80723451e-01],
           [  2.83622122e-02]])
    >>> D.se_betas(reg)
    array([ 18.69196608,   0.54769142,   0.19021045,   0.33120731])
    >>> reg=GM_Lag(y, X, w, w_lags=2, robust='white', name_x=['inc', 'crime'], name_y='hoval', name_ds='columbus')
    >>> reg.betas
    array([[  4.53017056e+01],
           [  6.20888617e-01],
           [ -4.80723451e-01],
           [  2.83622122e-02]])
    >>> D.se_betas(reg)
    array([ 26.46540238,   0.54128829,   0.20836574,   0.607184  ])
    >>> # instrument for HOVAL with DISCBD
    >>> X = np.array(db.by_col("INC"))
    >>> X = np.reshape(X, (49,1))
    >>> yd = np.array(db.by_col("CRIME"))
    >>> yd = np.reshape(yd, (49,1))
    >>> q = np.array(db.by_col("DISCBD"))
    >>> q = np.reshape(q, (49,1))
    >>> reg=GM_Lag(y, X, w, yd, q, w_lags=2, name_x=['inc'], name_y='hoval', name_yend=['crime'], name_q=['discbd'], name_ds='columbus')


    """
    def __init__(self, y, x, w, yend=None, q=None, w_lags=1,\
                    constant=True, robust=None, wk=None, nonspat_diag=True, name_y=None,
                    name_x=None, name_yend=None, name_q=None, name_ds=None,\
                    vm=False, pred=False, spat_lags='xq'):

        USER.check_arrays(y, x, yend, q)
        USER.check_weights(w, y)
        BaseGM_Lag.__init__(self, y=y, x=x, w=w, yend=yend, q=q,\
                            w_lags=w_lags, constant=constant, robust=robust,\
                            spat_lags=spat_lags)
        self.title = "SPATIAL TWO STAGE LEAST SQUARES"        
        self.name_ds = USER.set_name_ds(name_ds)
        self.name_y = USER.set_name_y(name_y)
        self.name_x = USER.set_name_x(name_x, x, constant)
        self.name_yend = USER.set_name_yend(name_yend, yend)
        self.name_yend.append(USER.set_name_yend_sp(self.name_y))
        self.name_z = self.name_x + self.name_yend
        self.name_q = USER.set_name_q(name_q, q)
        self.name_q.extend(USER.set_name_q_sp(self.name_x, w_lags))
        self.name_h = USER.set_name_h(self.name_x, self.name_q)
        self._get_diagnostics(w=w, beta_diag=True, nonspat_diag=nonspat_diag,\
                                    vm=vm, pred=pred)

    def _get_diagnostics(self, beta_diag=True, w=None, nonspat_diag=True,\
                              vm=False, pred=False):
        USER.DiagnosticBuilder.__init__(self, w=w, beta_diag=True,\
                                            nonspat_diag=nonspat_diag,\
                                            vm=vm, pred=pred, instruments=True)



def _test():
    import doctest
    doctest.testmod()

if __name__ == '__main__':
    _test()
    import numpy as np
    import pysal
    import pysal.spreg.diagnostics as D
    w = pysal.rook_from_shapefile("examples/columbus.shp")
    w.transform = 'r'
    db=pysal.open("examples/columbus.dbf","r")
    y = np.array(db.by_col("HOVAL"))
    y = np.reshape(y, (49,1))
    # no non-spatial endogenous variables
    X = []
    X.append(db.by_col("INC"))
    X.append(db.by_col("CRIME"))
    X = np.array(X).T
    reg=BaseGM_Lag(y, X, w, w_lags=2)
    print reg.betas
    print reg.vm
    reg=BaseGM_Lag(y, X, w, w_lags=2, robust='white')
    print reg.betas
    print reg.vm
    from robust import robust_vm
    print robust_vm(reg)
