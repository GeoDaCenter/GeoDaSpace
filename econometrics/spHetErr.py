import numpy as np
import numpy.linalg as la
import ols as OLS
import user_output as USER
import utils as GMM
import twosls as TSLS
from scipy import sparse as SP
from pysal import lag_spatial

class BaseGM_Error_Het:
    """
    GMM method for a spatial error model with heteroskedasticity (note: no
    consistency checks)

    Based on Arraiz et al [1]_

    ...

    Parameters
    ----------

    x           : array
                  nxk array with independent variables aligned with y
    y           : array
                  nx1 array with dependent variables
    w           : W
                  PySAL weights instance aligned with y and with instances S
                  and A1 created
    cycles      : int
                  Optional. Number of iterations of steps 2a. and 2b. Set to 1
                  by default

    Attributes
    ----------

    x           : array
                  nxk array of independent variables
    y           : array
                  nx1 array of dependent variable  
    betas       : array
                  (k+1)x1 array with estimates for betas and lambda
    n           : int
                  Number of observations
    k           : int
                  Number of variables (constant included)
    u           : array
                  nx1 array with residuals
    vm          : array
                  (k+1)x(k+1) variance-covariance matrix
    predy       : array
                  nx1 array of predicted values

    References
    ----------

    .. [1] Arraiz, I., Drukker, D. M., Kelejian, H., Prucha, I. R. (2010) "A
    Spatial Cliff-Ord-Type Model with Heteroskedastic Innovations: Small and
    Large Sample Results". Journal of Regional Science, Vol. 60, No. 2, pp.
    592-614.

    Examples
    --------
    >>> import numpy as np
    >>> import pysal
    >>> db=pysal.open("examples/columbus.dbf","r")
    >>> y = np.array(db.by_col("HOVAL"))
    >>> y = np.reshape(y, (49,1))
    >>> X = []
    >>> X.append(db.by_col("INC"))
    >>> X.append(db.by_col("CRIME"))
    >>> X = np.array(X).T
    >>> w = pysal.rook_from_shapefile("examples/columbus.shp")
    >>> w.transform = 'r'
    >>> reg = BaseGM_Error_Het(y, X, w)
    >>> print np.around(np.hstack((reg.betas,np.sqrt(reg.vm.diagonal()).reshape(4,1))),4)
    [[ 47.9963  11.479 ]
     [  0.7105   0.3681]
     [ -0.5588   0.1616]
     [  0.4118   0.168 ]]
    """

    def __init__(self,y,x,w,cycles=1,constant=True,step1c=True): 
        
        #1a. OLS --> \tilde{betas}
        ols = OLS.BaseOLS(y, x, constant=constant)
        self.x, self.y, self.n, self.k = ols.x, ols.y, ols.n, ols.k
        w.A1 = GMM.get_A1_het(w.sparse)

        #1b. GMM --> \tilde{\lambda1}
        moments = GMM._moments2eqs(w.A1, w.sparse, ols.u)
        lambda1 = GMM.optim_moments(moments)

        if step1c:
            #1c. GMM --> \tilde{\lambda2}
            sigma = get_psi_sigma(w, ols.u, lambda1)
            vc1 = get_vc_het(w, sigma)
            lambda2 = GMM.optim_moments(moments,vc1)
        else:
            lambda2 = lambda1 #Required to match Stata.
        lambda_i = [lambda2]

        for i in range(cycles):
            #2a. reg -->\hat{betas}
            xs = GMM.get_spFilter(w, lambda_i[-1], self.x)
            ys = GMM.get_spFilter(w, lambda_i[-1], self.y)
            ols_s = OLS.BaseOLS(ys, xs, constant=False)
            self.predy = np.dot(self.x, ols_s.betas)
            self.u = self.y - self.predy

            #2b. GMM --> \hat{\lambda}
            sigma_i = get_psi_sigma(w, self.u, lambda_i[-1])
            vc_i = get_vc_het(w, sigma_i)
            moments_i = GMM._moments2eqs(w.A1, w.sparse, self.u)
            lambda3 = GMM.optim_moments(moments_i, vc_i)
            lambda_i.append(lambda3)

        sigma = get_psi_sigma(w, self.u, lambda3)
        vc3 = get_vc_het(w, sigma)
        self.vm = get_vm_het(moments_i[0], lambda3, self, w, vc3)
        self.betas = np.vstack((ols_s.betas, lambda3))
        self._cache = {}

class GM_Error_Het(BaseGM_Error_Het):
    """
    GMM method for a spatial error model with heteroskedasticity
    Based on Arraiz et al [1]_

    ...

    Parameters
    ----------

    x           : array
                  nxk array with independent variables aligned with y
    y           : array
                  nx1 array with dependent variables
    w           : W
                  PySAL weights instance aligned with y and with instances S
                  and A1 created
    cycles      : int
                  Optional. Number of iterations of steps 2a. and 2b. Set to 1
                  by default
    name_ds     : string
                  dataset's name
    name_y      : string
                  Dependent variable's name
    name_x      : tuple
                  Independent variables' names

    Attributes
    ----------

    x           : array
                  nxk array of independent variables
    y           : array
                  nx1 array of dependent variable  
    betas       : array
                  (k+1)x1 array with estimates for betas and lambda
    n           : int
                  Number of observations
    k           : int
                  Number of variables (constant included)
    u           : array
                  nx1 array with residuals
    vm          : array
                  (k+1)x(k+1) variance-covariance matrix
    predy       : array
                  nx1 array of predicted values

    References
    ----------

    .. [1] Arraiz, I., Drukker, D. M., Kelejian, H., Prucha, I. R. (2010) "A
    Spatial Cliff-Ord-Type Model with Heteroskedastic Innovations: Small and
    Large Sample Results". Journal of Regional Science, Vol. 60, No. 2, pp.
    592-614.

    Examples
    --------
    >>> import numpy as np
    >>> import pysal
    >>> db=pysal.open("examples/columbus.dbf","r")
    >>> y = np.array(db.by_col("HOVAL"))
    >>> y = np.reshape(y, (49,1))
    >>> X = []
    >>> X.append(db.by_col("INC"))
    >>> X.append(db.by_col("CRIME"))
    >>> X = np.array(X).T
    >>> w = pysal.rook_from_shapefile("examples/columbus.shp")
    >>> w.transform = 'r'
    >>> reg = GM_Error_Het(y, X, w, name_y='home value', name_x=['income', 'crime'], name_ds='columbus')
    >>> print reg.name_x
    ['CONSTANT', 'income', 'crime', 'lambda']
    >>> print np.around(np.hstack((reg.betas,np.sqrt(reg.vm.diagonal()).reshape(4,1))),4)
    [[ 47.9963  11.479 ]
     [  0.7105   0.3681]
     [ -0.5588   0.1616]
     [  0.4118   0.168 ]]

    """

    def __init__(self, y, x, w, cycles=1, constant=True, step1c=True,\
                        name_y=None, name_x=None, name_ds=None):
        USER.check_arrays(y, x)
        USER.check_weights(w, y)
        BaseGM_Error_Het.__init__(self, y, x, w, cycles=cycles, constant=constant, step1c=step1c)
        self.title = "SPATIALLY WEIGHTED LEAST SQUARES"        
        self.name_ds = USER.set_name_ds(name_ds)
        self.name_y = USER.set_name_y(name_y)
        self.name_x = USER.set_name_x(name_x, x, constant)
        self.name_x.append('lambda')
        self.summary = str(np.around(np.hstack((self.betas,
               np.sqrt(self.vm.diagonal()).reshape(self.betas.shape[0],1))),4))
        

class BaseGM_Endog_Error_Het:
    """
    GMM method for a spatial error model with heteroskedasticity and
    endogenous variables (note: no consistency checks)

    Based on Arraiz et al [1]_

    ...

    Parameters
    ----------

    x           : array
                  nxk array with independent variables aligned with y
    y           : array
                  nx1 array with dependent variables
    yend        : array
                  Endogenous variables
    q           : array
                  array of instruments for yend (note: this should not contain
                  any variables from x;
    w           : W
                  PySAL weights instance aligned with y
    cycles      : int
                  Optional. Number of iterations of steps 2a. and 2b. Set to 1
                  by default

    Attributes
    ----------
    
    y           : array
                  nx1 array of dependent variable
    x           : array
                  array of independent variables (with constant added if
                  constant parameter set to True)
    z           : array
                  nxk array of variables (combination of x and yend)
    h           : array
                  nxl array of instruments (combination of x and q)
    yend        : array
                  endogenous variables
    q           : array
                  array of external exogenous variables
    betas       : array
                  (k+1)x1 array with estimates for betas and lambda
    u           : array
                  nx1 array of residuals 
    predy       : array
                  nx1 array of predicted values 
    n           : integer
                  number of observations
    k           : int
                  Number of variables, including exogenous and endogenous
                  variables and constant
    vm          : array
                  (k+1)x(k+1) variance-covariance matrix

    References
    ----------

    .. [1] Arraiz, I., Drukker, D. M., Kelejian, H., Prucha, I. R. (2010) "A
    Spatial Cliff-Ord-Type Model with Heteroskedastic Innovations: Small and
    Large Sample Results". Journal of Regional Science, Vol. 60, No. 2, pp.
    592-614.

    Examples
    --------
    >>> import numpy as np
    >>> import pysal
    >>> db=pysal.open("examples/columbus.dbf","r")
    >>> y = np.array(db.by_col("HOVAL"))
    >>> y = np.reshape(y, (49,1))
    >>> X = []
    >>> X.append(db.by_col("INC"))
    >>> X = np.array(X).T
    >>> yd = []
    >>> yd.append(db.by_col("CRIME"))
    >>> yd = np.array(yd).T
    >>> q = []
    >>> q.append(db.by_col("DISCBD"))
    >>> q = np.array(q).T
    >>> w = pysal.rook_from_shapefile("examples/columbus.shp")
    >>> w.transform = 'r'
    >>> reg = BaseGM_Endog_Error_Het(y, X, w, yd, q)
    >>> print np.around(np.hstack((reg.betas,np.sqrt(reg.vm.diagonal()).reshape(4,1))),4)
    [[ 55.3971  28.8901]
     [  0.4656   0.7731]
     [ -0.6704   0.468 ]
     [  0.4114   0.1777]]
    """

    def __init__(self,y,x,w,yend,q,cycles=1,constant=True,step1c=True): 
        #1a. reg --> \tilde{betas} 
        tsls = TSLS.BaseTSLS(y, x, yend, q=q, constant=constant)
        self.x, self.z, self.h, self.y = tsls.x, tsls.z, tsls.h, tsls.y
        self.yend, self.q, self.n, self.k = tsls.yend, tsls.q, tsls.n, tsls.k
        w.A1 = GMM.get_A1_het(w.sparse)

        #1b. GMM --> \tilde{\lambda1}
        moments = GMM._moments2eqs(w.A1, w.sparse, tsls.u)
        lambda1 = GMM.optim_moments(moments)

        if step1c:
            #1c. GMM --> \tilde{\lambda2}
            self.u = tsls.u
            zs = GMM.get_spFilter(w, lambda1, self.z)
            vc1 = get_vc_het_tsls(w, self, lambda1, tsls.pfora1a2, zs, filt=False)
            lambda2 = GMM.optim_moments(moments,vc1)
        else:
            lambda2 = lambda1 #Required to match Stata.
        lambda_i = [lambda2]

        for i in range(cycles):
            #2a. reg -->\hat{betas}
            xs = GMM.get_spFilter(w, lambda_i[-1], self.x)
            ys = GMM.get_spFilter(w, lambda_i[-1], self.y)
            yend_s = GMM.get_spFilter(w, lambda_i[-1], self.yend)
            tsls_s = TSLS.BaseTSLS(ys, xs, yend_s, h=self.h, constant=False)
            self.predy = np.dot(self.z, tsls_s.betas)
            self.u = self.y - self.predy

            #2b. GMM --> \hat{\lambda}
            vc2 = get_vc_het_tsls(w, self, lambda_i[-1], tsls_s.pfora1a2, np.hstack((xs,yend_s)))
            moments_i = GMM._moments2eqs(w.A1, w.sparse, self.u)
            lambda3 = GMM.optim_moments(moments_i, vc2)
            lambda_i.append(lambda3)

        zs = GMM.get_spFilter(w, lambda3, self.z)
        P = get_P_hat(self, tsls.hthi, zs)
        vc3 = get_vc_het_tsls(w, self, lambda3, P, zs, save_a1a2=True)
        self.vm = get_Omega_GS2SLS(w, lambda3, self, moments_i[0], vc3, P)
        self.betas = np.vstack((tsls_s.betas, lambda3))
        self._cache = {}


class GM_Endog_Error_Het(BaseGM_Endog_Error_Het):
    """
    GMM method for a spatial error model with heteroskedasticity and endogenous variables

    Based on Arraiz et al [1]_

    ...

    Parameters
    ----------

    x           : array
                  nxk array with independent variables aligned with y
    y           : array
                  nx1 array with dependent variables
    yend        : array
                  Endogenous variables
    q           : array
                  array of instruments for yend (note: this should not contain
                  any variables from x;
    w           : W
                  PySAL weights instance aligned with y
    cycles      : int
                  Optional. Number of iterations of steps 2a. and 2b. Set to 1
                  by default
    name_y      : string
                  Name of dependent variables for use in output
    name_x      : list of strings
                  Names of independent variables for use in output
    name_yend   : list of strings
                  Names of endogenous variables for use in output
    name_q      : list of strings
                  Names of instruments for use in output
    name_ds     : string
                  Name of dataset for use in output

    Attributes
    ----------
    
    y           : array
                  nx1 array of dependent variable
    x           : array
                  array of independent variables (with constant added if
                  constant parameter set to True)
    z           : array
                  nxk array of variables (combination of x and yend)
    h           : array
                  nxl array of instruments (combination of x and q)
    yend        : array
                  endogenous variables
    q           : array
                  array of external exogenous variables
    betas       : array
                  (k+1)x1 array with estimates for betas and lambda
    u           : array
                  nx1 array of residuals 
    predy       : array
                  nx1 array of predicted values 
    n           : integer
                  number of observations
    k           : int
                  Number of variables, including exogenous and endogenous
                  variables and constant
    vm          : array
                  (k+1)x(k+1) variance-covariance matrix

    References
    ----------

    .. [1] Arraiz, I., Drukker, D. M., Kelejian, H., Prucha, I. R. (2010) "A
    Spatial Cliff-Ord-Type Model with Heteroskedastic Innovations: Small and
    Large Sample Results". Journal of Regional Science, Vol. 60, No. 2, pp.
    592-614.


    Examples
    --------
    >>> import numpy as np
    >>> import pysal
    >>> db=pysal.open("examples/columbus.dbf","r")
    >>> y = np.array(db.by_col("HOVAL"))
    >>> y = np.reshape(y, (49,1))
    >>> X = []
    >>> X.append(db.by_col("INC"))
    >>> X = np.array(X).T
    >>> yd = []
    >>> yd.append(db.by_col("CRIME"))
    >>> yd = np.array(yd).T
    >>> q = []
    >>> q.append(db.by_col("DISCBD"))
    >>> q = np.array(q).T
    >>> w = pysal.rook_from_shapefile("examples/columbus.shp")
    >>> w.transform = 'r'
    >>> reg = GM_Endog_Error_Het(y, X, w, yd, q, name_x=['inc'], name_y='hoval', name_yend=['crime'], name_q=['discbd'], name_ds='columbus')
    >>> print reg.name_z
    ['CONSTANT', 'inc', 'crime', 'lambda']
    >>> print np.around(np.hstack((reg.betas,np.sqrt(reg.vm.diagonal()).reshape(4,1))),4)
    [[ 55.3971  28.8901]
     [  0.4656   0.7731]
     [ -0.6704   0.468 ]
     [  0.4114   0.1777]]

    """
    def __init__(self, y, x, w, yend, q, cycles=1, constant=True,\
                        step1c=True, name_y=None, name_x=None,\
                        name_yend=None, name_q=None, name_ds=None):
        USER.check_arrays(y, x, yend, q)
        USER.check_weights(w, y)
        BaseGM_Endog_Error_Het.__init__(self, y, x, w, yend, q, cycles=cycles,\
                                       constant=constant, step1c=step1c)
        self.title = "GENERALIZED SPATIAL TWO STAGE LEAST SQUARES"
        self.name_ds = USER.set_name_ds(name_ds)
        self.name_y = USER.set_name_y(name_y)
        self.name_x = USER.set_name_x(name_x, x, constant)
        self.name_yend = USER.set_name_yend(name_yend, yend)
        self.name_z = self.name_x + self.name_yend
        self.name_z.append('lambda')  #listing lambda last
        self.name_q = USER.set_name_q(name_q, q)
        self.name_h = USER.set_name_h(self.name_x, self.name_q)
        self.summary = str(np.around(np.hstack((self.betas,
               np.sqrt(self.vm.diagonal()).reshape(self.betas.shape[0],1))),4))
        

class BaseGM_Combo_Het(BaseGM_Endog_Error_Het):
    """
    GMM method for a spatial lag and error model with heteroskedasticity and
    endogenous variables  (note: no consistency checks) 

    Based on Arraiz et al [1]_

    ...

    Parameters
    ----------

    y           : array
                  nx1 array with dependent variable
    x           : array
                  nxk array with independent variables aligned with y
    w           : W
                  PySAL weights instance aligned with y
    yend        : array
                  Optional. Additional non-spatial endogenous variables (spatial lag is added by default)
    q           : array
                  array of instruments for yend (note: this should not contain
                  any variables from x; spatial instruments are computed by 
                  default)
    w_lags      : int
                  Number of orders to power W when including it as intrument
                  for the spatial lag (e.g. if w_lags=1, then the only
                  instrument is WX; if w_lags=2, the instrument is WWX; and so
                  on)    
    constant    : boolean
                  If true it appends a vector of ones to the independent variables
                  to estimate intercept (set to True by default)
    cycles      : int
                  Optional. Number of iterations of steps 2a. and 2b. Set to 1
                  by default

    Attributes
    ----------
    
    y           : array
                  nx1 array of dependent variable
    x           : array
                  array of independent variables (with constant added if
                  constant parameter set to True)
    z           : array
                  nxk array of variables (combination of x and yend)
    h           : array
                  nxl array of instruments (combination of x and q)
    yend        : array
                  endogenous variables
    q           : array
                  array of external exogenous variables
    betas       : array
                  (k+1)x1 array with estimates for betas and lambda
    u           : array
                  nx1 array of residuals 
    predy       : array
                  nx1 array of predicted values 
    n           : integer
                  number of observations
    k           : int
                  Number of variables, including exogenous and endogenous
                  variables and constant
    vm          : array
                  (k+1)x(k+1) variance-covariance matrix

    References
    ----------

    .. [1] Arraiz, I., Drukker, D. M., Kelejian, H., Prucha, I. R. (2010) "A
    Spatial Cliff-Ord-Type Model with Heteroskedastic Innovations: Small and
    Large Sample Results". Journal of Regional Science, Vol. 60, No. 2, pp.
    592-614.

    Examples
    --------
    >>> import numpy as np
    >>> import pysal
    >>> db=pysal.open("examples/columbus.dbf","r")
    >>> y = np.array(db.by_col("HOVAL"))
    >>> y = np.reshape(y, (49,1))
    >>> X = []
    >>> X.append(db.by_col("INC"))
    >>> X = np.array(X).T
    >>> w = pysal.rook_from_shapefile("examples/columbus.shp")
    >>> w.transform = 'r'

    Example only with spatial lag

    >>> reg = BaseGM_Combo_Het(y, X, w)
    >>> print np.around(np.hstack((reg.betas,np.sqrt(reg.vm.diagonal()).reshape(4,1))),4)
    [[  9.9753  14.1434]
     [  1.5742   0.374 ]
     [  0.1535   0.3978]
     [  0.2103   0.3924]]

    Example with both spatial lag and other endogenous variables

    >>> yd = []
    >>> yd.append(db.by_col("CRIME"))
    >>> yd = np.array(yd).T
    >>> q = []
    >>> q.append(db.by_col("DISCBD"))
    >>> q = np.array(q).T
    >>> reg = BaseGM_Combo_Het(y, X, w, yd, q)
    >>> betas = np.array([['CONSTANT'],['inc'],['crime'],['lag_hoval'],['lambda']])
    >>> print np.hstack((betas, np.around(np.hstack((reg.betas, np.sqrt(reg.vm.diagonal()).reshape(5,1))),5)))
    [['CONSTANT' '113.91292' '64.38815']
     ['inc' '-0.34822' '1.18219']
     ['crime' '-1.35656' '0.72482']
     ['lag_hoval' '-0.57657' '0.75856']
     ['lambda' '0.65608' '0.15719']]
    """

    def __init__(self, y, x, w, yend=None, q=None, w_lags=1,\
                    constant=True, cycles=1, step1c=True):
        # Create spatial lag of y
        yl = lag_spatial(w, y)
        if issubclass(type(yend), np.ndarray):  # spatial and non-spatial instruments
            lag_vars = np.hstack((x, q))
            spatial_inst = GMM.get_lags(w ,lag_vars, w_lags)
            q = np.hstack((q, spatial_inst))
            yend = np.hstack((yend, yl))
        elif yend == None:                   # spatial instruments only
            q = GMM.get_lags(w, x, w_lags)
            yend = yl
        else:
            raise Exception, "invalid value passed to yend"
        BaseGM_Endog_Error_Het.__init__(self, y, x, w, yend, q, cycles=cycles, constant=constant, step1c=step1c)

class GM_Combo_Het(BaseGM_Combo_Het):
    """
    GMM method for a spatial lag and error model with heteroskedasticity and
    endogenous variables  (note: no consistency checks) 

    Based on Arraiz et al [1]_

    ...

    Parameters
    ----------

    y           : array
                  nx1 array with dependent variable
    x           : array
                  nxk array with independent variables aligned with y
    w           : W
                  PySAL weights instance aligned with y
    yend        : array
                  Optional. Additional non-spatial endogenous variables (spatial lag is added by default)
    q           : array
                  array of instruments for yend (note: this should not contain
                  any variables from x; spatial instruments are computed by 
                  default)
    w_lags      : int
                  Number of orders to power W when including it as intrument
                  for the spatial lag (e.g. if w_lags=1, then the only
                  instrument is WX; if w_lags=2, the instrument is WWX; and so
                  on)    
    constant    : boolean
                  If true it appends a vector of ones to the independent variables
                  to estimate intercept (set to True by default)
    cycles      : int
                  Optional. Number of iterations of steps 2a. and 2b. Set to 1
                  by default
    name_y      : string
                  Name of dependent variables for use in output
    name_x      : list of strings
                  Names of independent variables for use in output
    name_yend   : list of strings
                  Names of endogenous variables for use in output
    name_q      : list of strings
                  Names of instruments for use in output
    name_ds     : string
                  Name of dataset for use in output

    Attributes
    ----------
    
    y           : array
                  nx1 array of dependent variable
    x           : array
                  array of independent variables (with constant added if
                  constant parameter set to True)
    z           : array
                  nxk array of variables (combination of x and yend)
    h           : array
                  nxl array of instruments (combination of x and q)
    yend        : array
                  endogenous variables
    q           : array
                  array of external exogenous variables
    betas       : array
                  (k+1)x1 array with estimates for betas and lambda
    u           : array
                  nx1 array of residuals 
    predy       : array
                  nx1 array of predicted values 
    n           : integer
                  number of observations
    k           : int
                  Number of variables, including exogenous and endogenous
                  variables and constant
    vm          : array
                  (k+1)x(k+1) variance-covariance matrix

    Examples
    --------
    >>> import numpy as np
    >>> import pysal
    >>> db=pysal.open("examples/columbus.dbf","r")
    >>> y = np.array(db.by_col("HOVAL"))
    >>> y = np.reshape(y, (49,1))
    >>> X = []
    >>> X.append(db.by_col("INC"))
    >>> X = np.array(X).T
    >>> w = pysal.rook_from_shapefile("examples/columbus.shp")
    >>> w.transform = 'r'

    Example only with spatial lag

    >>> reg = GM_Combo_Het(y, X, w, name_y='hoval', name_x=['income'], name_ds='columbus')
    >>> print reg.name_z
    ['CONSTANT', 'income', 'lag_hoval', 'lambda']
    >>> print np.around(np.hstack((reg.betas,np.sqrt(reg.vm.diagonal()).reshape(4,1))),4)
    [[  9.9753  14.1434]
     [  1.5742   0.374 ]
     [  0.1535   0.3978]
     [  0.2103   0.3924]]
        
    Example with both spatial lag and other endogenous variables

    >>> yd = []
    >>> yd.append(db.by_col("CRIME"))
    >>> yd = np.array(yd).T
    >>> q = []
    >>> q.append(db.by_col("DISCBD"))
    >>> q = np.array(q).T
    >>> reg = GM_Combo_Het(y, X, w, yd, q, name_x=['inc'], name_y='hoval', name_yend=['crime'], name_q=['discbd'], name_ds='columbus')
    >>> print reg.name_z
    ['CONSTANT', 'inc', 'crime', 'lag_hoval', 'lambda']
    >>> print np.round(reg.betas,4)
    [[ 113.9129]
     [  -0.3482]
     [  -1.3566]
     [  -0.5766]
     [   0.6561]]
    
    """
    
    def __init__(self, y, x, w, yend=None, q=None, w_lags=1,\
                    constant=True, cycles=1, step1c=True,\
                    name_y=None, name_x=None, name_yend=None,\
                    name_q=None, name_ds=None):

        USER.check_arrays(y, x, yend, q)
        USER.check_weights(w, y)
        BaseGM_Combo_Het.__init__(self, y, x, w, yend, q, w_lags,\
                                           constant, cycles, step1c)
        self.title = "GENERALIZED SPATIAL TWO STAGE LEAST SQUARES"        
        self.name_ds = USER.set_name_ds(name_ds)
        self.name_y = USER.set_name_y(name_y)
        self.name_x = USER.set_name_x(name_x, x, constant)
        self.name_yend = USER.set_name_yend(name_yend, yend)
        self.name_yend.append(USER.set_name_yend_sp(self.name_y))
        self.name_z = self.name_x + self.name_yend
        self.name_z.append('lambda')  #listing lambda last
        self.name_q = USER.set_name_q(name_q, q)
        self.name_q.extend(USER.set_name_q_sp(self.name_x, w_lags))
        self.name_h = USER.set_name_h(self.name_x, self.name_q)
        self.summary = str(np.around(np.hstack((self.betas,
               np.sqrt(self.vm.diagonal()).reshape(self.betas.shape[0],1))),4))

def get_psi_sigma(w, u, lamb):
    """
    Computes the Sigma matrix needed to compute Psi

    Parameters
    ----------
    w           : W
                  Spatial weights instance (requires 'S' and 'A1')

    u           : array
                  nx1 vector of residuals

    lamb        : float
                  Lambda

    """

    e = (u - lamb * (w.sparse * u)) ** 2
    E = SP.dia_matrix((e.flat,0), shape=(w.n,w.n))
    return E.tocsr()

def get_vc_het(w, E):
    """
    Computes the VC matrix Psi based on lambda as in Arraiz et al [1]_:

    ..math::

        \tilde{Psi} = \left(\begin{array}{c c}
                            \psi_{11} & \psi_{12} \\
                            \psi_{21} & \psi_{22} \\
                      \end{array} \right)

    NOTE: psi12=psi21

    ...

    Parameters
    ----------

    w           : W
                  Spatial weights instance (requires 'S' and 'A1')

    E           : sparse matrix
                  Sigma
 
    Returns
    -------

    Psi         : array
                  2x2 array with estimator of the variance-covariance matrix

    References
    ----------

    .. [1] Arraiz, I., Drukker, D. M., Kelejian, H., Prucha, I. R. (2010) "A
    Spatial Cliff-Ord-Type Model with Heteroskedastic Innovations: Small and
    Large Sample Results". Journal of Regional Science, Vol. 60, No. 2, pp.
    592-614.

    """
    aPatE = 2*w.A1* E
    wPwtE = (w.sparse + w.sparse.T) * E

    psi11 = aPatE * aPatE
    psi12 = aPatE * wPwtE
    psi22 = wPwtE * wPwtE 
    psi = map(np.sum, [psi11.diagonal(), psi12.diagonal(), psi22.diagonal()])
    return np.array([[psi[0], psi[1]], [psi[1], psi[2]]]) / (2. * w.n)

def get_vm_het(G, lamb, reg, w, psi):
    """
    Computes the variance-covariance matrix Omega as in Arraiz et al [1]_:
    ...

    Parameters
    ----------

    G           : array
                  G from moments equations

    lamb        : float
                  Final lambda from spHetErr estimation

    reg         : regression object
                  output instance from a regression model

    u           : array
                  nx1 vector of residuals

    w           : W
                  Spatial weights instance

    psi         : array
                  2x2 array with the variance-covariance matrix of the moment equations
 
    Returns
    -------

    vm          : array
                  (k+1)x(k+1) array with the variance-covariance matrix of the parameters

    References
    ----------

    .. [1] Arraiz, I., Drukker, D. M., Kelejian, H., Prucha, I. R. (2010) "A
    Spatial Cliff-Ord-Type Model with Heteroskedastic Innovations: Small and
    Large Sample Results". Journal of Regional Science, Vol. 60, No. 2, pp.
    592-614.

    """

    J = np.dot(G, np.array([[1],[2 * lamb]]))
    Zs = GMM.get_spFilter(w,lamb,reg.x)
    ZstEZs = np.dot((Zs.T * get_psi_sigma(w, reg.u, lamb)), Zs)
    ZsZsi = la.inv(np.dot(Zs.T,Zs))
    omega11 = w.n * np.dot(np.dot(ZsZsi,ZstEZs),ZsZsi)
    omega22 = la.inv(np.dot(np.dot(J.T,la.inv(psi)),J))
    zero = np.zeros((reg.k,1),float)
    vm = np.vstack((np.hstack((omega11, zero)),np.hstack((zero.T, omega22)))) / w.n
    return vm

def get_P_hat(reg, hthi, zf):
    """
    P_hat from Appendix B, used for a1 a2, using filtered Z
    """
    htzf = np.dot(reg.h.T, zf)
    P1 = np.dot(hthi, htzf)
    P2 = np.dot(htzf.T, P1)
    P2i = la.inv(P2)
    return reg.n*np.dot(P1, P2i)

def get_a1a2(w, reg, lambdapar, P, zs, filt):
    """
    Computes the a1 in psi assuming residuals come from original regression
    ...

    Parameters
    ----------

    w           : W
                  Spatial weights instance 

    reg         : TSLS
                  Two stage least quare regression instance
                  
    lambdapar   : float
                  Spatial autoregressive parameter
 
    Returns
    -------

    [a1, a2]    : list
                  a1 and a2 are two nx1 array in psi equation

    References
    ----------

    .. [1] Anselin, L. GMM Estimation of Spatial Error Autocorrelation with Heteroskedasticity
    
    """
    us = GMM.get_spFilter(w, lambdapar, reg.u)
    alpha1 = (-2.0/w.n) * (np.dot((zs.T * w.A1), us))
    alpha2 = (-1.0/w.n) * (np.dot((zs.T * (w.sparse + w.sparse.T)), us))
    a1 = np.dot(np.dot(reg.h, P), alpha1)
    a2 = np.dot(np.dot(reg.h, P), alpha2)
    if not filt:
        a1 = GMM.power_expansion(w, a1, lambdapar, transpose=True).T
        a2 = GMM.power_expansion(w, a2, lambdapar, transpose=True).T
    return [a1, a2]

def get_vc_het_tsls(w, reg, lambdapar, P, zs, filt=True, save_a1a2=False):

    sigma = get_psi_sigma(w, reg.u, lambdapar)
    vc1 = get_vc_het(w, sigma)
    a1, a2 = get_a1a2(w, reg, lambdapar, P, zs, filt)
    a1s = a1.T * sigma
    a2s = a2.T * sigma
    psi11 = float(np.dot(a1s, a1))
    psi12 = float(np.dot(a1s, a2))
    psi21 = float(np.dot(a2s, a1))
    psi22 = float(np.dot(a2s, a2))
    psi0 = np.array([[psi11, psi12], [psi21, psi22]]) / w.n
    if save_a1a2:
        psi = (vc1 + psi0, a1, a2)
    else:
        psi = vc1 + psi0
    return psi

def get_Omega_GS2SLS(w, lamb, reg, G, psi, P):
    """
    Computes the variance-covariance matrix for GS2SLS:
    ...

    Parameters
    ----------

    w           : W
                  Spatial weights instance 

    lamb        : float
                  Spatial autoregressive parameter
                  
    reg         : GSTSLS
                  Generalized Spatial two stage least quare regression instance
    G           : array
                  Moments
    psi         : array
                  Weighting matrix
 
    Returns
    -------

    omega       : array
                  (k+1)x(k+1)                 
    """
    psi, a1, a2 = psi
    sigma=get_psi_sigma(w, reg.u, lamb)
    psi_dd_1=(1.0/w.n) * reg.h.T * sigma 
    psi_dd = np.dot(psi_dd_1, reg.h)
    psi_dl=np.dot(psi_dd_1,np.hstack((a1,a2)))
    psi_o=np.hstack((np.vstack((psi_dd, psi_dl.T)), np.vstack((psi_dl, psi))))
    psii=la.inv(psi)
   
    j = np.dot(G, np.array([[1.], [2*lamb]]))
    jtpsii=np.dot(j.T, psii)
    jtpsiij=np.dot(jtpsii, j)
    jtpsiiji=la.inv(jtpsiij)
    omega_1=np.dot(jtpsiiji, jtpsii)
    omega_2=np.dot(np.dot(psii, j), jtpsiiji)
    om_1_s=omega_1.shape
    om_2_s=omega_2.shape
    p_s=P.shape
    omega_left=np.hstack((np.vstack((P.T, np.zeros((om_1_s[0],p_s[0])))), 
               np.vstack((np.zeros((p_s[1], om_1_s[1])), omega_1))))
    omega_right=np.hstack((np.vstack((P, np.zeros((om_2_s[0],p_s[1])))), 
               np.vstack((np.zeros((p_s[0], om_2_s[1])), omega_2))))
    omega=np.dot(np.dot(omega_left, psi_o), omega_right)    
    return omega / w.n
                    

def _test():
    import doctest
    doctest.testmod()

if __name__ == '__main__':
    _test()
    import numpy as np
    import pysal
    db=pysal.open("examples/columbus.dbf","r")
    y = np.array(db.by_col("CRIME"))
    y = np.reshape(y, (49,1))
    X = []
    X.append(db.by_col("INC"))
    X = np.array(X).T
    yd = np.array(db.by_col("HOVAL"))
    yd = np.reshape(yd, (49,1))
    q = np.array(db.by_col("DISCBD"))
    q = np.reshape(q, (49,1))
    w = pysal.rook_from_shapefile("examples/columbus.shp")
    w.transform = 'r'
    reg = GM_Error_Het(y, X, w)
    print "Exogenous variables only:"
    print "Dependent variable: CRIME"
    print "Variable  Coef.  S.E."
    print "Constant %5.4f %5.4f" % (reg.betas[0],np.sqrt(reg.vm.diagonal())[0])
    for i in range(len(reg.betas)-2):
        print "Var_%s %5.4f %5.4f" % (i+1,reg.betas[i+1],np.sqrt(reg.vm.diagonal())[i+1])
    print "Lambda: %5.4f %5.4f" % (reg.betas[-1],np.sqrt(reg.vm.diagonal())[-1])
    print '$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$'
    print "Spatial Lag:"
    reg = GM_Combo_Het(y, X, w, yd, q)
    print "Dependent variable: CRIME"
    print "Variable  Coef.  S.E."
    print "Constant %5.4f %5.4f" % (reg.betas[0],np.sqrt(reg.vm.diagonal())[0])
    for i in range(len(reg.betas)-2):
        print "Var_%s %5.4f %5.4f" % (i+1,reg.betas[i+1],np.sqrt(reg.vm.diagonal())[i+1])
    print "Lambda: %5.4f %5.4f" % (reg.betas[-1],np.sqrt(reg.vm.diagonal())[-1])
