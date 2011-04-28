import numpy as np
import numpy.linalg as la
import pysal.spreg.ols as OLS
import pysal.spreg.user_output as USER
import utils as GMM
import twosls as TSLS
from power_expansion import power_expansion
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
     [  0.4118  14.4391]]
    
    """

    def __init__(self,y,x,w,cycles=1,constant=True): ######Inserted i parameter here for iterations...
        #1a. OLS --> \tilde{betas}
        ols = OLS.BaseOLS(y, x, constant=constant)
        self.x = ols.x
        self.y = y
        self.n, self.k = ols.x.shape

        #1b. GMM --> \tilde{\lambda1}
        moments = moments_het(w, ols.u)
        lambda1 = GMM.optim_moments(moments)

        #1c. GMM --> \tilde{\lambda2}
        sigma = get_psi_sigma(w, ols.u, lambda1)
        vc1 = get_vc_het(w, sigma)
        lambda2 = GMM.optim_moments(moments,vc1)
        
        ols.betas, lambda3, vc2, G, ols.u = self.iterate(cycles,ols,w,lambda2)
        #Output
        self.betas = np.vstack((ols.betas,lambda3))
        self.vm = get_vm_het(G,lambda3,ols,w,vc2)
        self._cache = {}

        @property
        def predy(self):
            if 'predy' not in self._cache:
                self._cache['predy'] = np.dot(self.x,self.betas[0:-1])
            return self._cache['predy']

        @property
        def u(self):
            if 'u' not in self._cache:
                self._cache['u'] = self.y - self.predy
            return self._cache['u']

    def iterate(self,cycles,reg,w,lambda2):
        for n in range(cycles):
            #2a. reg -->\hat{betas}
            xs,ys = GMM.get_spFilter(w,lambda2,reg.x),GMM.get_spFilter(w,lambda2,reg.y)            
            beta_i = np.dot(np.linalg.inv(np.dot(xs.T,xs)),np.dot(xs.T,ys))
            predy = np.dot(reg.x, beta_i)
            u = reg.y - predy
            #2b. GMM --> \hat{\lambda}
            moments_i = moments_het(w, u)
            sigma_i =  get_psi_sigma(w, u, lambda2)
            vc2 = get_vc_het(w, sigma_i)
            lambda2 = GMM.optim_moments(moments_i,vc2)
        return beta_i,lambda2,vc2,moments_i[0], u

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
     [  0.4118  14.4391]]

    """

    def __init__(self, y, x, w, cycles=1, constant=True,\
                        name_y=None, name_x=None, name_ds=None):
        USER.check_arrays(y, x)
        USER.check_weights(w, y)
        BaseGM_Error_Het.__init__(self, y, x, w, cycles=cycles, constant=constant)
        self.title = "SPATIALLY WEIGHTED LEAST SQUARES"        
        self.name_ds = USER.set_name_ds(name_ds)
        self.name_y = USER.set_name_y(name_y)
        self.name_x = USER.set_name_x(name_x, x, constant)
        self.name_x.append('lambda')
        

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
    [[ 55.3971  22.9247]
     [  0.4656   0.6863]
     [ -0.6704   0.3504]
     [  0.4137  13.1586]]
    """

    def __init__(self,y,x,w,yend,q,cycles=1,constant=True): 
        #1a. OLS --> \tilde{betas} 
        tsls = TSLS.BaseTSLS(y, x, yend, q=q, constant=constant)
        self.x = tsls.x
        self.y = y
        self.yend = yend
        self.q = tsls.q
        self.n, self.k = tsls.x.shape

        #1b. GMM --> \tilde{\lambda1}
        moments = moments_het(w, tsls.u)
        lambda1 = GMM.optim_moments(moments)

        #1c. GMM --> \tilde{\lambda2}
        vc1 = get_vc_het_tsls(w, tsls, lambda1)
        lambda2 = GMM.optim_moments(moments,vc1)
        
        tsls.betas, lambda3, vc2, G, tsls.u = self.iterate(cycles,tsls,w,lambda2)
        self.u = tsls.u
        #Output
        self.betas = np.vstack((tsls.betas,lambda3))
        self.vm = get_Omega_GS2SLS(w, lambda3, tsls, G, vc2)
        self._cache = {}

        @property
        def predy(self):
            if 'predy' not in self._cache:
                self._cache['predy'] = np.dot(np.hstack((self.x,self.yend)),self.betas[0:-1])
            return self._cache['predy']

        @property
        def u(self):
            if 'u' not in self._cache:
                self._cache['u'] = self.y - self.predy
            return self._cache['u']

        @property
        def z(self):
            if 'z' not in self._cache:
                self._cache['z'] = np.hstack((self.x,self.yend))
            return self._cache['z']

        @property
        def h(self):
            if 'h' not in self._cache:
                self._cache['h'] = np.hstack((self.x,self.q))
            return self._cache['h']

    def iterate(self,cycles,reg,w,lambda2):
        for n in range(cycles):
            #2a. reg -->\hat{betas}
            xs,ys = GMM.get_spFilter(w,lambda2,reg.x),GMM.get_spFilter(w,lambda2,reg.y)
            yend_s = GMM.get_spFilter(w,lambda2, reg.yend)
            tsls = TSLS.BaseTSLS(ys, xs, yend_s, h=reg.h, constant=False)
            predy = np.dot(reg.z, tsls.betas)
            tsls.u = reg.y - predy
            #2b. GMM --> \hat{\lambda}
            moments_i = moments_het(w, tsls.u)
            vc2 = get_vc_het_tsls(w, tsls, lambda2)
            lambda2 = GMM.optim_moments(moments_i,vc2)
        return tsls.betas,lambda2,vc2,moments_i[0], tsls.u

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
    [[ 55.3971  22.9247]
     [  0.4656   0.6863]
     [ -0.6704   0.3504]
     [  0.4137  13.1586]]

    """
    def __init__(self, y, x, w, yend, q, cycles=1, constant=True,\
                        name_y=None, name_x=None, name_yend=None,\
                        name_q=None, name_ds=None):
        USER.check_arrays(y, x, yend, q)
        USER.check_weights(w, y)
        BaseGM_Endog_Error_Het.__init__(self, y, x, w, yend, q, cycles=cycles,\
                                       constant=constant)
        self.title = "GENERALIZED SPATIAL TWO STAGE LEAST SQUARES"
        self.name_ds = USER.set_name_ds(name_ds)
        self.name_y = USER.set_name_y(name_y)
        self.name_x = USER.set_name_x(name_x, x, constant)
        self.name_yend = USER.set_name_yend(name_yend, yend)
        self.name_z = self.name_x + self.name_yend
        self.name_z.append('lambda')  #listing lambda last
        self.name_q = USER.set_name_q(name_q, q)
        self.name_h = USER.set_name_h(self.name_x, self.name_q)
        


class BaseGM_Combo_Het(BaseGM_Endog_Error_Het):
    """
    GMM method for a spatial lang and error model with heteroskedasticity and
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
    [[   9.9753   11.8351]
     [   1.5742    0.3781]
     [   0.1535    0.3625]
     [   0.1984  122.1638]]

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
    [['CONSTANT' '113.91292' '43.19183']
     ['inc' '-0.34822' '0.89651']
     ['crime' '-1.35656' '0.49404']
     ['lag_hoval' '-0.57657' '0.47791']
     ['lambda' '0.65607' '27.01048']]
    """

    def __init__(self, y, x, w, yend=None, q=None, w_lags=1,\
                    constant=True, cycles=1):
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
        BaseGM_Endog_Error_Het.__init__(self, y, x, w, yend, q, cycles=cycles, constant=constant)

class GM_Combo_Het(BaseGM_Combo_Het):
    """
    GMM method for a spatial lang and error model with heteroskedasticity and
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
    [[   9.9753   11.8351]
     [   1.5742    0.3781]
     [   0.1535    0.3625]
     [   0.1984  122.1638]]
        
        
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
                    constant=True, cycles=1,\
                    name_y=None, name_x=None, name_yend=None,\
                    name_q=None, name_ds=None):

        USER.check_arrays(y, x, yend, q)
        USER.check_weights(w, y)
        BaseGM_Combo_Het.__init__(self, y, x, w, yend, q, w_lags,\
                                           constant, cycles)
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


def moments_het(w, u):
    """
    Function to compute all six components of the system of equations for a
    spatial error model with heteroskedasticity estimated by GMM as in Arraiz
    et al [1]_

    Scipy sparse matrix version. It implements eqs. A.1 in Appendix A of
    Arraiz et al. (2007) by using all matrix manipulation
    
    [g1] + [G11 G12] *  [\lambda]    = [0]
    [g2]   [G21 G22]    [\lambda^2]    [0]

    NOTE: 'residuals' has been renamed 'u' to fit paper notation

    ...
    
    Parameters
    ----------

    w           : W
                  Spatial weights instance (requires 'S' and 'A1')

    u           : array
                  Residuals. nx1 array assumed to be aligned with w
 
    Attributes
    ----------

    moments     : list
                  List of two arrays corresponding to the matrices 'G' and
                  'g', respectively.

    References
    ----------

    .. [1] Arraiz, I., Drukker, D. M., Kelejian, H., Prucha, I. R. (2010) "A
    Spatial Cliff-Ord-Type Model with Heteroskedastic Innovations: Small and
    Large Sample Results". Journal of Regional Science, Vol. 60, No. 2, pp.
    592-614.

    """
    ut = u.T
    S = w.sparse
    St = S.T
    A1 = GMM.get_A1(S)

    utSt = ut * St
    A1u = A1 * u
    Su = S * u

    g1 = np.dot(ut, A1u)
    g2 = np.dot(ut, Su)
    g = np.array([[g1][0][0],[g2][0][0]]) / w.n

    G11 = 2 * (np.dot(utSt, A1u)) 
    G12 = -np.dot(utSt * A1, Su)
    G21 = np.dot(utSt, ((S + St) * u))
    G22 = -np.dot(utSt, (S * Su))
    G = np.array([[G11[0][0],G12[0][0]],[G21[0][0],G22[0][0]]]) / w.n
    return [G, g]

def get_psi_sigma(w, u, l):
    """
    Computes the Sigma matrix needed to compute Psi

    Parameters
    ----------
    w           : W
                  Spatial weights instance (requires 'S' and 'A1')

    u           : array
                  nx1 vector of residuals

    l           : float
                  Lambda

    """

    e = (u - l * (w.sparse * u)) ** 2
    E = SP.lil_matrix(w.sparse.get_shape())
    E.setdiag(e.flat)
    E = E.asformat('csr')
    return E

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
    A1=GMM.get_A1(w.sparse)
    A1t = A1.T
    wt = w.sparse.T

    aPatE = (A1 + A1t) * E
    wPwtE = (w.sparse + wt) * E

    psi11 = aPatE * aPatE
    psi12 = aPatE * wPwtE
    psi22 = wPwtE * wPwtE 
    psi = map(np.sum, [psi11.diagonal(), psi12.diagonal(), psi22.diagonal()])
    return np.array([[psi[0], psi[1]], [psi[1], psi[2]]]) / (2 * w.n)

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

def get_vc_het_tsls(w, reg, lambdapar):

    sigma = get_psi_sigma(w, reg.u, lambdapar)
    vc1 = get_vc_het(w, sigma)
    a1, a2 = get_a1a2(w, reg, lambdapar)
    a1s = a1.T * sigma
    a2s = a2.T * sigma
    psi11 = float(np.dot(a1s, a1))
    psi12 = float(np.dot(a1s, a2))
    psi21 = float(np.dot(a2s, a1))
    psi22 = float(np.dot(a2s, a2))
    psi = np.array([[psi11, psi12], [psi21, psi22]]) / w.n
    return vc1 + psi

def get_Omega_GS2SLS(w, lamb, reg, G, psi):
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
    
    sigma=get_psi_sigma(w, reg.u, lamb)
    psi_dd_1=(1.0/w.n) * reg.h.T * sigma 
    psi_dd = np.dot(psi_dd_1, reg.h)
    a1a2=get_a1a2(w, reg, lamb)
    psi_dl=np.dot(psi_dd_1,np.hstack(tuple(a1a2)))
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
    p_s=reg.pfora1a2.shape
    
    omega_left=np.hstack((np.vstack((reg.pfora1a2.T, np.zeros((om_1_s[0],p_s[0])))), 
               np.vstack((np.zeros((p_s[1], om_1_s[1])), omega_1))))
    omega_right=np.hstack((np.vstack((reg.pfora1a2, np.zeros((om_2_s[0],p_s[1])))), 
               np.vstack((np.zeros((p_s[0], om_2_s[1])), omega_2))))
    omega=np.dot(np.dot(omega_left, psi_o), omega_right)    
    return omega / w.n

def get_a1a2(w,reg,lambdapar):
    """
    Computes the a1 in psi equation:
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
    zst = GMM.get_spFilter(w,lambdapar, reg.z).T
    us = GMM.get_spFilter(w,lambdapar, reg.u)
    alpha1 = (-2.0/w.n) * (np.dot((zst * GMM.get_A1(w.sparse)), us))
    alpha2 = (-1.0/w.n) * (np.dot((zst * (w.sparse + w.sparse.T)), us))
    v1 = np.dot(np.dot(reg.h, reg.pfora1a2), alpha1)
    v2 = np.dot(np.dot(reg.h, reg.pfora1a2), alpha2)
    a1t = power_expansion(w, v1, lambdapar, transpose=True)
    a2t = power_expansion(w, v2, lambdapar, transpose=True)
    return [a1t.T, a2t.T]

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
    yd = []
    yd.append(db.by_col("HOVAL"))
    yd = np.array(yd).T
    q = []
    q.append(db.by_col("DISCBD"))
    q = np.array(q).T
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
