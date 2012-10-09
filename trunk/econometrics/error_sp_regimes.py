import numpy as np
from numpy import linalg as la
from pysal import lag_spatial
from econometrics.ols import BaseOLS
from econometrics.twosls import BaseTSLS
from econometrics.error_sp import BaseGM_Error, BaseGM_Endog_Error, _momentsGM_Error
from utils import power_expansion, set_endog, iter_msg, sp_att
from utils import get_A1_hom, get_A2_hom, get_A1_het, optim_moments, get_spFilter, get_lags, _moments2eqs
from utils import spdot, RegressionPropsY
import regimes as REGI
import user_output as USER
import summary_output as SUMMARY

class GM_Error_Regimes(RegressionPropsY, REGI.Regimes_Frame):
    """
    GMM method for a spatial error model with regimes, with results and diagnostics;
    based on Kelejian and Prucha (1998, 1999)[1]_ [2]_.

    Parameters
    ----------
    y            : array
                   nx1 array for dependent variable
    x            : array
                   Two dimensional array with n rows and one column for each
                   independent (exogenous) variable, excluding the constant
    regimes      : list
                   List of n values with the mapping of each
                   observation to a regime. Assumed to be aligned with 'x'.
    w            : pysal W object
                   Spatial weights object (note: if provided then spatial
                   diagnostics are computed)   
    constant_regi: [False, 'one', 'many']
                   Switcher controlling the constant term setup. It may take
                   the following values:
                     *  'one': a vector of ones is appended to x and held
                               constant across regimes
                     * 'many': a vector of ones is appended to x and considered
                               different per regime (default)
    cols2regi    : list, 'all'
                   Argument indicating whether each
                   column of x should be considered as different per regime
                   or held constant across regimes (False).
                   If a list, k booleans indicating for each variable the
                   option (True if one per regime, False to be held constant).
                   If 'all' (default), all the variables vary by regime.
    regime_error : boolean
                   If True, the spatial parameter for autoregressive error is also
                   computed according to different regimes. If False (default), 
                   the spatial parameter is fixed accross regimes.
    vm           : boolean
                   If True, include variance-covariance matrix in summary
                   results
    name_y       : string
                   Name of dependent variable for use in output
    name_x       : list of strings
                   Names of independent variables for use in output
    name_w       : string
                   Name of weights matrix for use in output
    name_ds      : string
                   Name of dataset for use in output
    name_regimes : string
                   Name of regime variable for use in the output


    Attributes
    ----------
    summary      : string
                   Summary of regression results and diagnostics (note: use in
                   conjunction with the print command)
    betas        : array
                   kx1 array of estimated coefficients
    u            : array
                   nx1 array of residuals
    e_filtered   : array
                   nx1 array of spatially filtered residuals
    predy        : array
                   nx1 array of predicted y values
    n            : integer
                   Number of observations
    k            : integer
                   Number of variables for which coefficients are estimated
                   (including the constant)
    y            : array
                   nx1 array for dependent variable
    x            : array
                   Two dimensional array with n rows and one column for each
                   independent (exogenous) variable, including the constant
    mean_y       : float
                   Mean of dependent variable
    std_y        : float
                   Standard deviation of dependent variable
    pr2          : float
                   Pseudo R squared (squared correlation between y and ypred)
    vm           : array
                   Variance covariance matrix (kxk)
    sig2         : float
                   Sigma squared used in computations
    std_err      : array
                   1xk array of standard errors of the betas    
    z_stat       : list of tuples
                   z statistic; each tuple contains the pair (statistic,
                   p-value), where each is a float
    name_y       : string
                   Name of dependent variable for use in output
    name_x       : list of strings
                   Names of independent variables for use in output
    name_w       : string
                   Name of weights matrix for use in output
    name_ds      : string
                   Name of dataset for use in output
    name_regimes : string
                   Name of regime variable for use in the output
    title        : string
                   Name of the regression method used
    regimes      : list
                   List of n values with the mapping of each
                   observation to a regime. Assumed to be aligned with 'x'.
    constant_regi: ['one', 'many']
                   Ignored if regimes=False. Constant option for regimes.
                   Switcher controlling the constant term setup. It may take
                   the following values:                    
                     *  'one': a vector of ones is appended to x and held
                               constant across regimes
                     * 'many': a vector of ones is appended to x and considered
                               different per regime
    cols2regi    : list, 'all'
                   Ignored if regimes=False. Argument indicating whether each
                   column of x should be considered as different per regime
                   or held constant across regimes (False).
                   If a list, k booleans indicating for each variable the
                   option (True if one per regime, False to be held constant).
                   If 'all', all the variables vary by regime.
    regime_error : boolean
                   If True, the spatial parameter for autoregressive error is also
                   computed according to different regimes. If False (default), 
                   the spatial parameter is fixed accross regimes.
    kr           : int
                   Number of variables/columns to be "regimized" or subject
                   to change by regime. These will result in one parameter
                   estimate by regime for each variable (i.e. nr parameters per
                   variable)
    kf           : int
                   Number of variables/columns to be considered fixed or
                   global across regimes and hence only obtain one parameter
                   estimate
    nr           : int
                   Number of different regimes in the 'regimes' list

    References
    ----------

    .. [1] Kelejian, H.R., Prucha, I.R. (1998) "A generalized spatial
    two-stage least squares procedure for estimating a spatial autoregressive
    model with autoregressive disturbances". The Journal of Real State
    Finance and Economics, 17, 1.

    .. [2] Kelejian, H.R., Prucha, I.R. (1999) "A Generalized Moments
    Estimator for the Autoregressive Parameter in a Spatial Model".
    International Economic Review, 40, 2.

    Examples
    --------

    We first need to import the needed modules, namely numpy to convert the
    data we read into arrays that ``spreg`` understands and ``pysal`` to
    perform all the analysis.

    >>> import pysal
    >>> import numpy as np

    Open data on NCOVR US County Homicides (3085 areas) using pysal.open().
    This is the DBF associated with the NAT shapefile.  Note that
    pysal.open() also reads data in CSV format; since the actual class
    requires data to be passed in as numpy arrays, the user can read their
    data in using any method.  

    >>> db = pysal.open(pysal.examples.get_path("NAT.dbf"),'r')
    
    Extract the HR90 column (homicide rates in 1990) from the DBF file and make it the
    dependent variable for the regression. Note that PySAL requires this to be
    an numpy array of shape (n, 1) as opposed to the also common shape of (n, )
    that other packages accept.

    >>> y_var = 'HR90'
    >>> y = np.array([db.by_col(y_var)]).reshape(3085,1)
    
    Extract UE90 (unemployment rate) and PS90 (population structure) vectors from
    the DBF to be used as independent variables in the regression. Other variables
    can be inserted by adding their names to x_var, such as x_var = ['Var1','Var2','...]
    Note that PySAL requires this to be an nxj numpy array, where j is the
    number of independent variables (not including a constant). By default
    this model adds a vector of ones to the independent variables passed in.

    >>> x_var = ['PS90','UE90']
    >>> x = np.array([db.by_col(name) for name in x_var]).T

    The different regimes in this data are given according to the North and 
    South dummy (SOUTH).

    >>> r_var = 'SOUTH'
    >>> regimes = db.by_col(r_var)

    Since we want to run a spatial error model, we need to specify
    the spatial weights matrix that includes the spatial configuration of the
    observations. To do that, we can open an already existing gal file or 
    create a new one. In this case, we will create one from ``NAT.shp``.

    >>> w = pysal.rook_from_shapefile(pysal.examples.get_path("NAT.shp"))

    Unless there is a good reason not to do it, the weights have to be
    row-standardized so every row of the matrix sums to one. Among other
    things, this allows to interpret the spatial lag of a variable as the
    average value of the neighboring observations. In PySAL, this can be
    easily performed in the following way:

    >>> w.transform = 'r'

    We are all set with the preliminaries, we are good to run the model. In this
    case, we will need the variables and the weights matrix. If we want to
    have the names of the variables printed in the output summary, we will
    have to pass them in as well, although this is optional.

    >>> model = GM_Error_Regimes(y, x, regimes, w, name_y=y_var, name_x=x_var, name_regimes=r_var, name_ds='NAT.dbf')

    Once we have run the model, we can explore a little bit the output. The
    regression object we have created has many attributes so take your time to
    discover them. Note that because we are running the classical GMM error
    model from 1998/99, the spatial parameter is obtained as a point estimate, so
    although you get a value for it (there are for coefficients under
    model.betas), you cannot perform inference on it (there are only three
    values in model.se_betas). Alternatively, we can have a summary of the
    output by typing: model.summary
    
    >>> print model.name_x
    ['0_CONSTANT', '0_PS90', '0_UE90', '1_CONSTANT', '1_PS90', '1_UE90', 'lambda']
    >>> np.around(model.betas, decimals=6)
    array([[ 0.074807],
           [ 0.786107],
           [ 0.538849],
           [ 5.103756],
           [ 1.196009],
           [ 0.600533],
           [ 0.364103]])
    >>> np.around(model.std_err, decimals=6)
    array([ 0.379864,  0.152316,  0.051942,  0.471285,  0.19867 ,  0.057252])
    >>> np.around(model.z_stat, decimals=6)
    array([[  0.196932,   0.843881],
           [  5.161042,   0.      ],
           [ 10.37397 ,   0.      ],
           [ 10.829455,   0.      ],
           [  6.02007 ,   0.      ],
           [ 10.489215,   0.      ]])
    >>> np.around(model.sig2, decimals=6)
    28.172732

    """
    def __init__(self, y, x, regimes, w,\
                 vm=False, name_y=None, name_x=None, name_w=None,\
                 constant_regi='many', cols2regi='all', regime_error=False,\
                 name_ds=None, name_regimes=None):

        n = USER.check_arrays(y, x)
        USER.check_y(y, n)
        USER.check_weights(w, y)
        self.constant_regi = constant_regi
        self.cols2regi = cols2regi
        self.regime_error = regime_error

        x_constant = USER.check_constant(x)
        name_x = USER.set_name_x(name_x, x)
        self.name_x_r = name_x

        if constant_regi=='many':
            regi_cons = [True]
        elif constant_regi=='one':
            regi_cons = [False]
        if cols2regi=='all':
            cols2regi = regi_cons + [True]*x.shape[1]
        else:
            cols2regi = regi_cons + cols2regi

        self.x, self.name_x = REGI.Regimes_Frame.__init__(self, x_constant, \
                regimes, constant_regi=None, cols2regi=cols2regi, names=name_x)
        ols = BaseOLS(y=y, x=self.x)
        self.n, self.k = ols.x.shape
        self.y = ols.y
        """
        if regime_error == True:
            self.regimes_set = list(set(regimes))
            self.regimes_set.sort()
            w = REGI.w_regimes(w, regimes, self.regimes_set)
        """    
        moments = _momentsGM_Error(w, ols.u)
        lambda1 = optim_moments(moments)
        xs = get_spFilter(w, lambda1, x_constant)
        ys = get_spFilter(w, lambda1, y)
        xs = REGI.Regimes_Frame.__init__(self, xs,\
                regimes, constant_regi=None, cols2regi=cols2regi)[0]
        ols2 = BaseOLS(y=ys, x=xs)

        #Output
        self.predy = spdot(self.x, ols2.betas)
        self.u = y - self.predy
        self.betas = np.vstack((ols2.betas, np.array([[lambda1]])))
        self.sig2 = ols2.sig2n
        self.e_filtered = self.u - lambda1*lag_spatial(w,self.u)
        self.vm = self.sig2 * ols2.xtxi
        self.title = "SPATIALLY WEIGHTED LEAST SQUARES - REGIMES"        
        self.name_ds = USER.set_name_ds(name_ds)
        self.name_y = USER.set_name_y(name_y)
        self.name_x.append('lambda')
        self.name_w = USER.set_name_w(name_w, w)
        self.name_regimes = USER.set_name_ds(name_regimes)
        self.chow = REGI.Chow(self)
        self._cache = {}
        
        SUMMARY.GM_Error(reg=self, w=w, vm=vm, regimes=True)



class GM_Endog_Error_Regimes(RegressionPropsY, REGI.Regimes_Frame):
    '''
    GMM method for a spatial error model with regimes and endogenous variables, with
    results and diagnostics; based on Kelejian and Prucha (1998, 1999)[1]_[2]_.

    Parameters
    ----------
    y            : array
                   nx1 array for dependent variable
    x            : array
                   Two dimensional array with n rows and one column for each
                   independent (exogenous) variable, excluding the constant
    yend         : array
                   Two dimensional array with n rows and one column for each
                   endogenous variable
    q            : array
                   Two dimensional array with n rows and one column for each
                   external exogenous variable to use as instruments (note: 
                   this should not contain any variables from x)
    regimes      : list
                   List of n values with the mapping of each
                   observation to a regime. Assumed to be aligned with 'x'.
    w            : pysal W object
                   Spatial weights object (note: if provided then spatial
                   diagnostics are computed)   
    constant_regi: [False, 'one', 'many']
                   Switcher controlling the constant term setup. It may take
                   the following values:
                     *  'one': a vector of ones is appended to x and held
                               constant across regimes
                     * 'many': a vector of ones is appended to x and considered
                               different per regime (default)
    cols2regi    : list, 'all'
                   Argument indicating whether each
                   column of x should be considered as different per regime
                   or held constant across regimes (False).
                   If a list, k booleans indicating for each variable the
                   option (True if one per regime, False to be held constant).
                   If 'all' (default), all the variables vary by regime.
    regime_error : boolean
                   If True, the spatial parameter for autoregressive error is also
                   computed according to different regimes. If False (default), 
                   the spatial parameter is fixed accross regimes.
    vm           : boolean
                   If True, include variance-covariance matrix in summary
                   results
    name_y       : string
                   Name of dependent variable for use in output
    name_x       : list of strings
                   Names of independent variables for use in output
    name_yend    : list of strings
                   Names of endogenous variables for use in output
    name_q       : list of strings
                   Names of instruments for use in output
    name_w       : string
                   Name of weights matrix for use in output
    name_ds      : string
                   Name of dataset for use in output
    name_regimes : string
                   Name of regime variable for use in the output

    Attributes
    ----------
    summary      : string
                   Summary of regression results and diagnostics (note: use in
                   conjunction with the print command)
    betas        : array
                   kx1 array of estimated coefficients
    u            : array
                   nx1 array of residuals
    e_filtered   : array
                   nx1 array of spatially filtered residuals
    predy        : array
                   nx1 array of predicted y values
    n            : integer
                   Number of observations
    k            : integer
                   Number of variables for which coefficients are estimated
                   (including the constant)
    y            : array
                   nx1 array for dependent variable
    x            : array
                   Two dimensional array with n rows and one column for each
                   independent (exogenous) variable, including the constant
    yend         : array
                   Two dimensional array with n rows and one column for each
                   endogenous variable
    z            : array
                   nxk array of variables (combination of x and yend)
    mean_y       : float
                   Mean of dependent variable
    std_y        : float
                   Standard deviation of dependent variable
    vm           : array
                   Variance covariance matrix (kxk)
    pr2          : float
                   Pseudo R squared (squared correlation between y and ypred)
    sig2         : float
                   Sigma squared used in computations
    std_err      : array
                   1xk array of standard errors of the betas    
    z_stat       : list of tuples
                   z statistic; each tuple contains the pair (statistic,
                   p-value), where each is a float
    name_y        : string
                    Name of dependent variable for use in output
    name_x        : list of strings
                    Names of independent variables for use in output
    name_yend     : list of strings
                    Names of endogenous variables for use in output
    name_z        : list of strings
                    Names of exogenous and endogenous variables for use in 
                    output
    name_q        : list of strings
                    Names of external instruments
    name_h        : list of strings
                    Names of all instruments used in ouput
    name_w        : string
                    Name of weights matrix for use in output
    name_ds       : string
                    Name of dataset for use in output
    title         : string
                    Name of the regression method used
    regimes       : list
                    List of n values with the mapping of each
                    observation to a regime. Assumed to be aligned with 'x'.
    constant_regi : [False, 'one', 'many']
                    Ignored if regimes=False. Constant option for regimes.
                    Switcher controlling the constant term setup. It may take
                    the following values:
                      *  'one': a vector of ones is appended to x and held
                                constant across regimes
                      * 'many': a vector of ones is appended to x and considered
                                different per regime
    cols2regi     : list, 'all'
                    Ignored if regimes=False. Argument indicating whether each
                    column of x should be considered as different per regime
                    or held constant across regimes (False).
                    If a list, k booleans indicating for each variable the
                    option (True if one per regime, False to be held constant).
                    If 'all', all the variables vary by regime.
    regime_error : boolean
                   If True, the spatial parameter for autoregressive error is also
                   computed according to different regimes. If False (default), 
                   the spatial parameter is fixed accross regimes.
    kr            : int
                    Number of variables/columns to be "regimized" or subject
                    to change by regime. These will result in one parameter
                    estimate by regime for each variable (i.e. nr parameters per
                    variable)
    kf            : int
                    Number of variables/columns to be considered fixed or
                    global across regimes and hence only obtain one parameter
                    estimate
    nr            : int
                    Number of different regimes in the 'regimes' list

    References
    ----------

    .. [1] Kelejian, H.R., Prucha, I.R. (1998) "A generalized spatial
    two-stage least squares procedure for estimating a spatial autoregressive
    model with autoregressive disturbances". The Journal of Real State
    Finance and Economics, 17, 1.

    .. [2] Kelejian, H.R., Prucha, I.R. (1999) "A Generalized Moments
    Estimator for the Autoregressive Parameter in a Spatial Model".
    International Economic Review, 40, 2.

    Examples
    --------

    We first need to import the needed modules, namely numpy to convert the
    data we read into arrays that ``spreg`` understands and ``pysal`` to
    perform all the analysis.

    >>> import pysal
    >>> import numpy as np

    Open data on NCOVR US County Homicides (3085 areas) using pysal.open().
    This is the DBF associated with the NAT shapefile.  Note that
    pysal.open() also reads data in CSV format; since the actual class
    requires data to be passed in as numpy arrays, the user can read their
    data in using any method.  

    >>> db = pysal.open(pysal.examples.get_path("NAT.dbf"),'r')
    
    Extract the HR90 column (homicide rates in 1990) from the DBF file and make it the
    dependent variable for the regression. Note that PySAL requires this to be
    an numpy array of shape (n, 1) as opposed to the also common shape of (n, )
    that other packages accept.

    >>> y_var = 'HR90'
    >>> y = np.array([db.by_col(y_var)]).reshape(3085,1)
    
    Extract UE90 (unemployment rate) and PS90 (population structure) vectors from
    the DBF to be used as independent variables in the regression. Other variables
    can be inserted by adding their names to x_var, such as x_var = ['Var1','Var2','...]
    Note that PySAL requires this to be an nxj numpy array, where j is the
    number of independent variables (not including a constant). By default
    this model adds a vector of ones to the independent variables passed in.

    >>> x_var = ['PS90','UE90']
    >>> x = np.array([db.by_col(name) for name in x_var]).T

    For the endogenous models, we add the endogenous variable RD90 (resource deprivation)
    and we decide to instrument for it with FP89 (families below poverty):

    >>> yd_var = ['RD90']
    >>> yend = np.array([db.by_col(name) for name in yd_var]).T
    >>> q_var = ['FP89']
    >>> q = np.array([db.by_col(name) for name in q_var]).T

    The different regimes in this data are given according to the North and 
    South dummy (SOUTH).

    >>> r_var = 'SOUTH'
    >>> regimes = db.by_col(r_var)

    Since we want to run a spatial error model, we need to specify the spatial
    weights matrix that includes the spatial configuration of the observations
    into the error component of the model. To do that, we can open an already 
    existing gal file or create a new one. In this case, we will create one 
    from ``NAT.shp``.

    >>> w = pysal.rook_from_shapefile(pysal.examples.get_path("NAT.shp"))

    Unless there is a good reason not to do it, the weights have to be
    row-standardized so every row of the matrix sums to one. Among other
    things, this allows to interpret the spatial lag of a variable as the
    average value of the neighboring observations. In PySAL, this can be
    easily performed in the following way:

    >>> w.transform = 'r'

    We are all set with the preliminaries, we are good to run the model. In this
    case, we will need the variables (exogenous and endogenous), the
    instruments and the weights matrix. If we want to
    have the names of the variables printed in the output summary, we will
    have to pass them in as well, although this is optional.

    >>> model = GM_Endog_Error_Regimes(y, x, yend, q, regimes, w, name_y=y_var, name_x=x_var, name_yend=yd_var, name_q=q_var, name_regimes=r_var, name_ds='NAT.dbf')

    Once we have run the model, we can explore a little bit the output. The
    regression object we have created has many attributes so take your time to
    discover them. Note that because we are running the classical GMM error
    model from 1998/99, the spatial parameter is obtained as a point estimate, so
    although you get a value for it (there are for coefficients under
    model.betas), you cannot perform inference on it (there are only three
    values in model.se_betas). Also, this regression uses a two stage least
    squares estimation method that accounts for the endogeneity created by the
    endogenous variables included. Alternatively, we can have a summary of the
    output by typing: model.summary

    >>> print model.name_z
    ['0_CONSTANT', '0_PS90', '0_UE90', '1_CONSTANT', '1_PS90', '1_UE90', '0_RD90', '1_RD90', 'lambda']
    >>> np.around(model.betas, decimals=5)
    array([[ 3.59718],
           [ 1.0652 ],
           [ 0.15822],
           [ 9.19754],
           [ 1.88082],
           [-0.24878],
           [ 2.46161],
           [ 3.57943],
           [ 0.25564]])
    >>> np.around(model.std_err, decimals=6)
    array([ 0.522633,  0.137555,  0.063054,  0.473654,  0.18335 ,  0.072786,
            0.300711,  0.240413])
    
    '''
    def __init__(self, y, x, yend, q, regimes, w,\
                 vm=False, constant_regi='many', cols2regi='all',\
                 regime_error=False, name_y=None, name_x=None,\
                 name_yend=None, name_q=None, name_w=None,\
                 name_ds=None, name_regimes=None, summ=True):      
        
        n = USER.check_arrays(y, x, yend, q)
        USER.check_y(y, n)
        USER.check_weights(w, y)
        self.constant_regi = constant_regi
        self.cols2regi = cols2regi
        
        x_constant = USER.check_constant(x)
        name_x = USER.set_name_x(name_x, x)
        if summ:
            name_yend = USER.set_name_yend(name_yend, yend)
            self.name_y = USER.set_name_y(name_y)
            name_q = USER.set_name_q(name_q, q)
        self.name_x_r = name_x + name_yend

        if constant_regi=='many':
            regi_cons = [True]
        elif constant_regi=='one':
            regi_cons = [False]
        if cols2regi=='all':
            cols2regi = regi_cons + [True]*(x.shape[1]+yend.shape[1])
        else:
            cols2regi = regi_cons + cols2regi

        q, name_q = REGI.Regimes_Frame.__init__(self, q,\
                regimes, constant_regi=None, cols2regi='all', names=name_q)
        x, name_x = REGI.Regimes_Frame.__init__(self, x_constant,\
                regimes, constant_regi=None, cols2regi=cols2regi,\
                names=name_x)
        yend2, name_yend = REGI.Regimes_Frame.__init__(self, yend,\
                regimes, constant_regi=None,\
                cols2regi=cols2regi, yend=True, names=name_yend)

        tsls = BaseTSLS(y=y, x=x, yend=yend2, q=q)
        self.n, self.k = tsls.z.shape
        self.x, self.y = tsls.x, tsls.y
        self.yend, self.z = tsls.yend, tsls.z
        moments = _momentsGM_Error(w, tsls.u)
        lambda1 = optim_moments(moments)
        xs = get_spFilter(w, lambda1, x_constant)
        xs = REGI.Regimes_Frame.__init__(self, xs,\
                regimes, constant_regi=None, cols2regi=cols2regi)[0]
        ys = get_spFilter(w, lambda1, y)
        yend_s = get_spFilter(w, lambda1, yend)
        yend_s = REGI.Regimes_Frame.__init__(self, yend_s,\
                regimes, constant_regi=None, cols2regi=cols2regi,\
                yend=True)[0]
        tsls2 = BaseTSLS(ys, xs, yend_s, h=tsls.h)

        #Output
        self.betas = np.vstack((tsls2.betas, np.array([[lambda1]])))
        self.predy = spdot(tsls.z, tsls2.betas)
        self.u = y - self.predy
        self.sig2 = float(np.dot(tsls2.u.T,tsls2.u)) / self.n
        self.e_filtered = self.u - lambda1*lag_spatial(w,self.u)
        self.vm = self.sig2 * tsls2.varb
        self.name_ds = USER.set_name_ds(name_ds)
        self.name_x = USER.set_name_x(name_x, x, constant=True)
        self.name_yend = USER.set_name_yend(name_yend, yend)
        self.name_z = self.name_x + self.name_yend
        self.name_z.append('lambda')
        self.name_q = USER.set_name_q(name_q, q)
        self.name_regimes = USER.set_name_ds(name_regimes)
        self.name_h = USER.set_name_h(self.name_x, self.name_q)
        self.name_w = USER.set_name_w(name_w, w)
        self.chow = REGI.Chow(self)
        self._cache = {}
        if summ:
            self.title = "SPATIALLY WEIGHTED TWO STAGE LEAST SQUARES - REGIMES"
            SUMMARY.GM_Endog_Error(reg=self, w=w, vm=vm, regimes=True)

class GM_Combo_Regimes(GM_Endog_Error_Regimes, REGI.Regimes_Frame):
    """
    GMM method for a spatial lag and error model with regimes and endogenous
    variables, with results and diagnostics; based on Kelejian and Prucha (1998,
    1999)[1]_[2]_.

    Parameters
    ----------
    y            : array
                   nx1 array for dependent variable
    x            : array
                   Two dimensional array with n rows and one column for each
                   independent (exogenous) variable, excluding the constant
    regimes      : list
                   List of n values with the mapping of each
                   observation to a regime. Assumed to be aligned with 'x'.
    yend         : array
                   Two dimensional array with n rows and one column for each
                   endogenous variable
    q            : array
                   Two dimensional array with n rows and one column for each
                   external exogenous variable to use as instruments (note: 
                   this should not contain any variables from x)
    w            : pysal W object
                   Spatial weights object (note: if provided then spatial
                   diagnostics are computed)   
    constant_regi: [False, 'one', 'many']
                   Switcher controlling the constant term setup. It may take
                   the following values:
                    
                     *  False: no constant term is appended in any way
                     *  'one': a vector of ones is appended to x and held
                               constant across regimes
                     * 'many': a vector of ones is appended to x and considered
                               different per regime (default)
    cols2regi    : list, 'all'
                   Argument indicating whether each
                   column of x should be considered as different per regime
                   or held constant across regimes (False).
                   If a list, k booleans indicating for each variable the
                   option (True if one per regime, False to be held constant).
                   If 'all' (default), all the variables vary by regime.
    regime_error : boolean
                   If True, the spatial parameter for autoregressive error is also
                   computed according to different regimes. If False (default), 
                   the spatial parameter is fixed accross regimes.
    regime_lag   : boolean
                   If True, the spatial parameter for spatial lag is also
                   computed according to different regimes. If False (default), 
                   the spatial parameter is fixed accross regimes.
    w_lags       : integer
                   Orders of W to include as instruments for the spatially
                   lagged dependent variable. For example, w_lags=1, then
                   instruments are WX; if w_lags=2, then WX, WWX; and so on.
    lag_q        : boolean
                   If True, then include spatial lags of the additional 
                   instruments (q).
    vm           : boolean
                   If True, include variance-covariance matrix in summary
                   results
    name_y       : string
                   Name of dependent variable for use in output
    name_x       : list of strings
                   Names of independent variables for use in output
    name_yend    : list of strings
                   Names of endogenous variables for use in output
    name_q       : list of strings
                   Names of instruments for use in output
    name_w       : string
                   Name of weights matrix for use in output
    name_ds      : string
                   Name of dataset for use in output
    name_regimes : string
                   Name of regime variable for use in the output

    Attributes
    ----------
    summary      : string
                   Summary of regression results and diagnostics (note: use in
                   conjunction with the print command)
    betas        : array
                   kx1 array of estimated coefficients
    u            : array
                   nx1 array of residuals
    e_filtered   : array
                   nx1 array of spatially filtered residuals
    e_pred       : array
                   nx1 array of residuals (using reduced form)
    predy        : array
                   nx1 array of predicted y values
    predy_e      : array
                   nx1 array of predicted y values (using reduced form)
    n            : integer
                   Number of observations
    k            : integer
                   Number of variables for which coefficients are estimated
                   (including the constant)
    y            : array
                   nx1 array for dependent variable
    x            : array
                   Two dimensional array with n rows and one column for each
                   independent (exogenous) variable, including the constant
    yend         : array
                   Two dimensional array with n rows and one column for each
                   endogenous variable
    z            : array
                   nxk array of variables (combination of x and yend)
    mean_y       : float
                   Mean of dependent variable
    std_y        : float
                   Standard deviation of dependent variable
    vm           : array
                   Variance covariance matrix (kxk)
    pr2          : float
                   Pseudo R squared (squared correlation between y and ypred)
    pr2_e        : float
                   Pseudo R squared (squared correlation between y and ypred_e
                   (using reduced form))
    sig2         : float
                   Sigma squared used in computations (based on filtered
                   residuals)
    std_err      : array
                   1xk array of standard errors of the betas    
    z_stat       : list of tuples
                   z statistic; each tuple contains the pair (statistic,
                   p-value), where each is a float
    name_y        : string
                    Name of dependent variable for use in output
    name_x        : list of strings
                    Names of independent variables for use in output
    name_yend     : list of strings
                    Names of endogenous variables for use in output
    name_z        : list of strings
                    Names of exogenous and endogenous variables for use in 
                    output
    name_q        : list of strings
                    Names of external instruments
    name_h        : list of strings
                    Names of all instruments used in ouput
    name_w        : string
                    Name of weights matrix for use in output
    name_ds       : string
                    Name of dataset for use in output
    title         : string
                    Name of the regression method used
    regimes       : list
                    List of n values with the mapping of each
                    observation to a regime. Assumed to be aligned with 'x'.
    constant_regi : [False, 'one', 'many']
                    Ignored if regimes=False. Constant option for regimes.
                    Switcher controlling the constant term setup. It may take
                    the following values:
                      *  'one': a vector of ones is appended to x and held
                                constant across regimes
                      * 'many': a vector of ones is appended to x and considered
                                different per regime
    cols2regi     : list, 'all'
                    Ignored if regimes=False. Argument indicating whether each
                    column of x should be considered as different per regime
                    or held constant across regimes (False).
                    If a list, k booleans indicating for each variable the
                    option (True if one per regime, False to be held constant).
                    If 'all', all the variables vary by regime.
    regime_error  : boolean
                    If True, the spatial parameter for autoregressive error is also
                    computed according to different regimes. If False (default), 
                    the spatial parameter is fixed accross regimes.
    regime_lag    : boolean
                    If True, the spatial parameter for spatial lag is also
                    computed according to different regimes. If False (default), 
                    the spatial parameter is fixed accross regimes.
    kr            : int
                    Number of variables/columns to be "regimized" or subject
                    to change by regime. These will result in one parameter
                    estimate by regime for each variable (i.e. nr parameters per
                    variable)
    kf            : int
                    Number of variables/columns to be considered fixed or
                    global across regimes and hence only obtain one parameter
                    estimate
    nr            : int
                    Number of different regimes in the 'regimes' list

    References
    ----------

    .. [1] Kelejian, H.R., Prucha, I.R. (1998) "A generalized spatial
    two-stage least squares procedure for estimating a spatial autoregressive
    model with autoregressive disturbances". The Journal of Real State
    Finance and Economics, 17, 1.

    .. [2] Kelejian, H.R., Prucha, I.R. (1999) "A Generalized Moments
    Estimator for the Autoregressive Parameter in a Spatial Model".
    International Economic Review, 40, 2.

    Examples
    --------

    We first need to import the needed modules, namely numpy to convert the
    data we read into arrays that ``spreg`` understands and ``pysal`` to
    perform all the analysis.

    >>> import numpy as np
    >>> import pysal

    Open data on NCOVR US County Homicides (3085 areas) using pysal.open().
    This is the DBF associated with the NAT shapefile.  Note that
    pysal.open() also reads data in CSV format; since the actual class
    requires data to be passed in as numpy arrays, the user can read their
    data in using any method.  

    >>> db = pysal.open(pysal.examples.get_path("NAT.dbf"),'r')
    
    Extract the HR90 column (homicide rates in 1990) from the DBF file and make it the
    dependent variable for the regression. Note that PySAL requires this to be
    an numpy array of shape (n, 1) as opposed to the also common shape of (n, )
    that other packages accept.

    >>> y_var = 'HR90'
    >>> y = np.array([db.by_col(y_var)]).reshape(3085,1)
    
    Extract UE90 (unemployment rate) and PS90 (population structure) vectors from
    the DBF to be used as independent variables in the regression. Other variables
    can be inserted by adding their names to x_var, such as x_var = ['Var1','Var2','...]
    Note that PySAL requires this to be an nxj numpy array, where j is the
    number of independent variables (not including a constant). By default
    this model adds a vector of ones to the independent variables passed in.

    >>> x_var = ['PS90','UE90']
    >>> x = np.array([db.by_col(name) for name in x_var]).T

    The different regimes in this data are given according to the North and 
    South dummy (SOUTH).

    >>> r_var = 'SOUTH'
    >>> regimes = db.by_col(r_var)

    Since we want to run a spatial lag model, we need to specify
    the spatial weights matrix that includes the spatial configuration of the
    observations. To do that, we can open an already existing gal file or 
    create a new one. In this case, we will create one from ``NAT.shp``.

    >>> w = pysal.rook_from_shapefile(pysal.examples.get_path("NAT.shp"))

    Unless there is a good reason not to do it, the weights have to be
    row-standardized so every row of the matrix sums to one. Among other
    things, this allows to interpret the spatial lag of a variable as the
    average value of the neighboring observations. In PySAL, this can be
    easily performed in the following way:

    >>> w.transform = 'r'

    The Combo class runs an SARAR model, that is a spatial lag+error model.
    In this case we will run a simple version of that, where we have the
    spatial effects as well as exogenous variables. Since it is a spatial
    model, we have to pass in the weights matrix. If we want to
    have the names of the variables printed in the output summary, we will
    have to pass them in as well, although this is optional.

    >>> model = GM_Combo_Regimes(y, x, regimes, w=w, name_y=y_var, name_x=x_var, name_regimes=r_var, name_ds='NAT')

    Once we have run the model, we can explore a little bit the output. The
    regression object we have created has many attributes so take your time to
    discover them. Note that because we are running the classical GMM error
    model from 1998/99, the spatial parameter is obtained as a point estimate, so
    although you get a value for it (there are for coefficients under
    model.betas), you cannot perform inference on it (there are only three
    values in model.se_betas). Also, this regression uses a two stage least
    squares estimation method that accounts for the endogeneity created by the
    spatial lag of the dependent variable. We can have a summary of the
    output by typing: model.summary 
    Alternatively, we can check the betas:

    >>> print model.name_z
    ['0_CONSTANT', '0_PS90', '0_UE90', '1_CONSTANT', '1_PS90', '1_UE90', 'Global_W_HR90', 'lambda']
    >>> print np.around(model.betas,4)
    [[ 1.4607]
     [ 0.958 ]
     [ 0.5658]
     [ 9.113 ]
     [ 1.1338]
     [ 0.6517]
     [-0.4583]
     [ 0.6136]]

    And lambda:

    >>> print 'lambda: ', np.around(model.betas[-1], 4)
    lambda:  [ 0.6136]
        
    This class also allows the user to run a spatial lag+error model with the
    extra feature of including non-spatial endogenous regressors. This means
    that, in addition to the spatial lag and error, we consider some of the
    variables on the right-hand side of the equation as endogenous and we
    instrument for this. In this case we consider RD90 (resource deprivation)
    as an endogenous regressor.  We use FP89 (families below poverty)
    for this and hence put it in the instruments parameter, 'q'.

    >>> yd_var = ['RD90']
    >>> yd = np.array([db.by_col(name) for name in yd_var]).T
    >>> q_var = ['FP89']
    >>> q = np.array([db.by_col(name) for name in q_var]).T

    And then we can run and explore the model analogously to the previous combo:

    >>> model = GM_Combo_Regimes(y, x, regimes, yd, q, w, name_y=y_var, name_x=x_var, name_yend=yd_var, name_q=q_var, name_regimes=r_var, name_ds='NAT')
    >>> print model.name_z
    ['0_CONSTANT', '0_PS90', '0_UE90', '1_CONSTANT', '1_PS90', '1_UE90', '0_RD90', '1_RD90', 'Global_W_HR90', 'lambda']
    >>> print model.betas
    [[ 3.41963782]
     [ 1.04065841]
     [ 0.16634393]
     [ 8.86544628]
     [ 1.85120528]
     [-0.24908469]
     [ 2.43014046]
     [ 3.61645481]
     [ 0.03308671]
     [ 0.18684992]]
    >>> print np.sqrt(model.vm.diagonal())
    [ 0.53067577  0.13271426  0.06058025  0.76406411  0.17969783  0.07167421
      0.28943121  0.25308326  0.06126529]
    >>> print 'lambda: ', np.around(model.betas[-1], 4)
    lambda:  [ 0.1868]
    """
    def __init__(self, y, x, regimes, yend=None, q=None,\
                 w=None, w_lags=1, lag_q=True,\
                 constant_regi='many', cols2regi='all',\
                 regime_error=False, regime_lag=False,\
                 vm=False, name_y=None, name_x=None,\
                 name_yend=None, name_q=None,\
                 name_w=None, name_ds=None, name_regimes=None):

        n = USER.check_arrays(y, x)
        USER.check_y(y, n)
        USER.check_weights(w, y)
        name_x = USER.set_name_x(name_x, x,constant=True)
        self.name_y = USER.set_name_y(name_y)
        name_yend = USER.set_name_yend(name_yend, yend)
        name_q = USER.set_name_q(name_q, q)
        name_q.extend(USER.set_name_q_sp(name_x, w_lags, name_q, lag_q, force_all=True))        

        if cols2regi == 'all':
            if yend!=None:
                cols2regi = [True] * (x.shape[1]+yend.shape[1])
            else:
                cols2regi = [True] * (x.shape[1])     

        if regime_lag == True:
            cols2regi += [True]
            self.regimes_set = list(set(regimes))
            self.regimes_set.sort()
            w_i,regi_ids = REGI.w_regimes(w, regimes, self.regimes_set, transform=regime_error, get_ids=regime_error)
            if not regime_error:
                w = REGI.w_regimes_union(w, w_i, self.regimes_set)
        else:
            cols2regi += [False]

        if regime_lag == True and regime_error == True:
            """
            if set(cols2regi) == set([True]):
                self.name_regimes = USER.set_name_ds(name_regimes)
                self.constant_regi=constant_regi
                self.cols2regi = cols2regi
                self.GM_Endog_Error_Regimes(y, x, w_i, regi_ids,\
                 yend=yend, q=q, w_lags=w_lags, lag_q=lag_q, cores=cores,\
                 robust=robust, gwk=gwk, sig2n_k=sig2n_k, cols2regi=cols2regi,\
                 spat_diag=spat_diag, vm=vm, name_y=name_y, name_x=name_x,\
                 name_yend=name_yend, name_q=name_q, name_regimes=self.name_regimes,\
                 name_w=name_w, name_gwk=name_gwk, name_ds=name_ds)
            else:
                raise Exception, "All coefficients must vary accross regimes if regime_error = True."
            """
            pass
        else:
            yend2, q2 = set_endog(y, x, w, yend, q, w_lags, lag_q)
            name_yend.append(USER.set_name_yend_sp(self.name_y))

            GM_Endog_Error_Regimes.__init__(self, y=y, x=x, yend=yend2,\
                    q=q2, regimes=regimes, w=w, vm=vm, constant_regi=constant_regi,\
                    cols2regi=cols2regi, regime_error=regime_error,\
                    name_y=self.name_y, name_x=name_x,\
                    name_yend=name_yend, name_q=name_q, name_w=name_w,\
                    name_ds=name_ds, name_regimes=name_regimes, summ=False)

            self.predy_e, self.e_pred = sp_att(w,self.y,\
                       self.predy,yend2[:,-1].reshape(self.n,1),self.betas[-2])
            self.regime_lag=regime_lag
            self.title = "SPATIALLY WEIGHTED TWO STAGE LEAST SQUARES - REGIMES"
            SUMMARY.GM_Combo(reg=self, w=w, vm=vm, regimes=True)

def _test():
    import doctest
    start_suppress = np.get_printoptions()['suppress']
    np.set_printoptions(suppress=True)    
    doctest.testmod()
    np.set_printoptions(suppress=start_suppress)

if __name__ == '__main__':

    _test()
    import pysal
    import numpy as np
    dbf = pysal.open(pysal.examples.get_path('columbus.dbf'),'r')
    y = np.array([dbf.by_col('CRIME')]).T
    names_to_extract = ['INC', 'HOVAL']
    x = np.array([dbf.by_col(name) for name in names_to_extract]).T
    regimes = regimes = dbf.by_col('NSA')
    w = pysal.open(pysal.examples.get_path("columbus.gal"), 'r').read() 
    w.transform='r'
    model = GM_Error_Regimes(y, x, regimes, w=w, name_y='crime', name_x=['income', 'hoval'], name_regimes='nsa', name_ds='columbus')
    print model.summary
    
