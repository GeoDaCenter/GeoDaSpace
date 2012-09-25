"""
Spatial Error Models module
"""

__author__ = "Luc Anselin luc.anselin@asu.edu, \
        Daniel Arribas-Bel darribas@asu.edu, \
        Pedro V. Amaral pedro.amaral@asu.edu"

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

    Open data on Columbus neighborhood crime (49 areas) using pysal.open().
    This is the DBF associated with the Columbus shapefile.  Note that
    pysal.open() also reads data in CSV format; since the actual class
    requires data to be passed in as numpy arrays, the user can read their
    data in using any method.  

    >>> db = pysal.open(pysal.examples.get_path('columbus.dbf'),'r')
    
    Extract the CRIME column (crime rates) from the DBF file and make it the
    dependent variable for the regression. Note that PySAL requires this to be
    an numpy array of shape (n, 1) as opposed to the also common shape of (n, )
    that other packages accept.

    >>> y_var = 'CRIME'
    >>> y = np.array([db.by_col(y_var)]).reshape(49,1)

    Extract HOVAL (house value) and INC (income) vectors from the DBF to be used as
    independent variables in the regression.  Note that PySAL requires this to
    be an nxj numpy array, where j is the number of independent variables (not
    including a constant). By default this class adds a vector of ones to the
    independent variables passed in.

    >>> x_var = ['INC','HOVAL']
    >>> x = np.array([db.by_col(name) for name in x_var]).T

    The different regimes in this data are given according to the North and 
    South dummy (NSA).

    >>> r_var = 'NSA'
    >>> regimes = db.by_col(r_var)

    Since we want to run a spatial error model, we need to specify the spatial
    weights matrix that includes the spatial configuration of the observations
    into the error component of the model. To do that, we can open an already
    existing gal file or create a new one. In this case, we will use
    ``columbus.gal``, which contains contiguity relationships between the
    observations in the Columbus dataset we are using throughout this example.
    Note that, in order to read the file, not only to open it, we need to
    append '.read()' at the end of the command.

    >>> w = pysal.open(pysal.examples.get_path("columbus.gal"), 'r').read() 
    
    Unless there is a good reason not to do it, the weights have to be
    row-standardized so every row of the matrix sums to one. Among other
    things, his allows to interpret the spatial lag of a variable as the
    average value of the neighboring observations. In PySAL, this can be
    easily performed in the following way:

    >>> w.transform='r'

    We are all set with the preliminars, we are good to run the model. In this
    case, we will need the variables and the weights matrix. If we want to
    have the names of the variables printed in the output summary, we will
    have to pass them in as well, although this is optional.

    >>> model = GM_Error_Regimes(y, x, regimes, w, name_y=y_var, name_x=x_var, name_regimes=r_var, name_ds='columbus')

    Once we have run the model, we can explore a little bit the output. The
    regression object we have created has many attributes so take your time to
    discover them. Note that because we are running the classical GMM error
    model from 1998/99, the spatial parameter is obtained as a point estimate, so
    although you get a value for it (there are for coefficients under
    model.betas), you cannot perform inference on it (there are only three
    values in model.se_betas). Alternatively, we can have a summary of the
    output by typing: model.summary
    
    >>> print model.name_x
    ['0.0_CONSTANT', '0.0_INC', '0.0_HOVAL', '1.0_CONSTANT', '1.0_INC', '1.0_HOVAL', 'lambda']
    >>> np.around(model.betas, decimals=6)
    array([[ 63.344307],
           [ -1.521865],
           [ -0.15468 ],
           [ 61.400714],
           [ -0.850761],
           [ -0.335501],
           [  0.386716]])
    >>> np.around(model.std_err, decimals=6)
    array([ 7.110468,  0.584779,  0.218793,  7.505965,  0.57366 ,  0.108007])
    >>> np.around(model.z_stat, decimals=6)
    array([[ 8.908599,  0.      ],
           [-2.602464,  0.009256],
           [-0.70697 ,  0.479585],
           [ 8.180256,  0.      ],
           [-1.483041,  0.138063],
           [-3.106292,  0.001894]])
    >>> np.around(model.sig2, decimals=6)
    102.130506

    """
    def __init__(self, y, x, regimes, w,\
                 vm=False, name_y=None, name_x=None, constant_regi='many',\
                 cols2regi='all', regime_error=False, name_w=None,\
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

    Open data on Columbus neighborhood crime (49 areas) using pysal.open().
    This is the DBF associated with the Columbus shapefile.  Note that
    pysal.open() also reads data in CSV format; since the actual class
    requires data to be passed in as numpy arrays, the user can read their
    data in using any method.  

    >>> db = pysal.open(pysal.examples.get_path("columbus.dbf"),'r')
    
    Extract the CRIME column (crime rates) from the DBF file and make it the
    dependent variable for the regression. Note that PySAL requires this to be
    an numpy array of shape (n, 1) as opposed to the also common shape of (n, )
    that other packages accept.

    >>> y_var = 'CRIME'
    >>> y = np.array([db.by_col(y_var)]).reshape(49,1)
    
    Extract INC (income) vector from the DBF to be used as
    independent variables in the regression.  Note that PySAL requires this to
    be an nxj numpy array, where j is the number of independent variables (not
    including a constant). By default this model adds a vector of ones to the
    independent variables passed in.

    >>> x_var = ['INC']
    >>> x = np.array([db.by_col(name) for name in x_var]).T

    In this case we consider HOVAL (home value) is an endogenous regressor.
    We tell the model that this is so by passing it in a different parameter
    from the exogenous variables (x).

    >>> yd_var = ['HOVAL']
    >>> yend = np.array([db.by_col(name) for name in yd_var]).T

    Because we have endogenous variables, to obtain a correct estimate of the
    model, we need to instrument for HOVAL. We use DISCBD (distance to the
    CBD) for this and hence put it in the instruments parameter, 'q'.

    >>> q_var = ['DISCBD']
    >>> q = np.array([db.by_col(name) for name in q_var]).T

    The different regimes in this data are given according to the North and 
    South dummy (NSA).

    >>> r_var = 'NSA'
    >>> regimes = db.by_col(r_var)

    Since we want to run a spatial error model, we need to specify the spatial
    weights matrix that includes the spatial configuration of the observations
    into the error component of the model. To do that, we can open an already
    existing gal file or create a new one. In this case, we will use
    ``columbus.gal``, which contains contiguity relationships between the
    observations in the Columbus dataset we are using throughout this example.
    Note that, in order to read the file, not only to open it, we need to
    append '.read()' at the end of the command.

    >>> w = pysal.open(pysal.examples.get_path("columbus.gal"), 'r').read() 

    Unless there is a good reason not to do it, the weights have to be
    row-standardized so every row of the matrix sums to one. Among other
    things, this allows to interpret the spatial lag of a variable as the
    average value of the neighboring observations. In PySAL, this can be
    easily performed in the following way:

    >>> w.transform='r'

    We are all set with the preliminars, we are good to run the model. In this
    case, we will need the variables (exogenous and endogenous), the
    instruments and the weights matrix. If we want to
    have the names of the variables printed in the output summary, we will
    have to pass them in as well, although this is optional.

    >>> model = GM_Endog_Error_Regimes(y, x, yend, q, regimes, w, name_y=y_var, name_x=x_var, name_yend=yd_var, name_q=q_var, name_regimes=r_var, name_ds='columbus')

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
    ['0.0_CONSTANT', '0.0_INC', '1.0_CONSTANT', '1.0_INC', '0.0_HOVAL', '1.0_HOVAL', 'lambda']
    >>> np.around(model.betas, decimals=5)
    array([[ 77.48384],
           [  4.52986],
           [ 78.93207],
           [  0.42186],
           [ -3.23824],
           [ -1.14758],
           [  0.20222]])
    >>> np.around(model.std_err, decimals=6)
    array([ 19.770744,   6.076668,  24.322543,   2.177768,   2.970783,
             0.943924])
    
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

    Open data on Columbus neighborhood crime (49 areas) using pysal.open().
    This is the DBF associated with the Columbus shapefile.  Note that
    pysal.open() also reads data in CSV format; since the actual class
    requires data to be passed in as numpy arrays, the user can read their
    data in using any method.  

    >>> db = pysal.open(pysal.examples.get_path("columbus.dbf"),'r')
    
    Extract the CRIME column (crime rates) from the DBF file and make it the
    dependent variable for the regression. Note that PySAL requires this to be
    an numpy array of shape (n, 1) as opposed to the also common shape of (n, )
    that other packages accept.

    >>> y_var = 'CRIME'
    >>> y = np.array([db.by_col(y_var)]).reshape(49,1)
    
    Extract INC (income) and HOVAL (house value) vector from the DBF to be used as
    independent variables in the regression.  Note that PySAL requires this to
    be an nxj numpy array, where j is the number of independent variables (not
    including a constant). By default this model adds a vector of ones to the
    independent variables passed in.

    >>> x_var = ['INC','HOVAL']
    >>> x = np.array([db.by_col(name) for name in x_var]).T

    The different regimes in this data are given according to the North and 
    South dummy (NSA).

    >>> r_var = 'NSA'
    >>> regimes = db.by_col(r_var)

    Since we want to run a spatial error model, we need to specify the spatial
    weights matrix that includes the spatial configuration of the observations
    into the error component of the model. To do that, we can open an already
    existing gal file or create a new one. In this case, we will create one
    from ``columbus.shp``.

    >>> w = pysal.rook_from_shapefile(pysal.examples.get_path("columbus.shp"))

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

    >>> model = GM_Combo_Regimes(y, x, regimes, w=w, name_y=y_var, name_x=x_var, name_regimes=r_var, name_ds='columbus')

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
    ['0.0_CONSTANT', '0.0_INC', '0.0_HOVAL', '1.0_CONSTANT', '1.0_INC', '1.0_HOVAL', 'Global_W_CRIME', 'lambda']
    >>> print np.around(model.betas,4)
    [[ 47.3659]
     [ -1.3584]
     [ -0.151 ]
     [ 43.8943]
     [ -0.8813]
     [ -0.2804]
     [  0.4159]
     [  0.0791]]

    And lambda:

    >>> print 'lambda: ', np.around(model.betas[-1], 4)
    lambda:  [ 0.0791]
        
    This class also allows the user to run a spatial lag+error model with the
    extra feature of including non-spatial endogenous regressors. This means
    that, in addition to the spatial lag and error, we consider some of the
    variables on the right-hand side of the equation as endogenous and we
    instrument for this. As an example, we will include HOVAL (home value) as
    endogenous and will instrument with DISCBD (distance to the CSB). We first
    need to read in the variables:

    >>> x_var = ['INC']
    >>> x = np.array([db.by_col(name) for name in x_var]).T
    >>> yd_var = ['HOVAL']
    >>> yend = np.array([db.by_col(name) for name in yd_var]).T
    >>> q_var = ['DISCBD']
    >>> q = np.array([db.by_col(name) for name in q_var]).T

    And then we can run and explore the model analogously to the previous combo:

    >>> model = GM_Combo_Regimes(y, x, regimes, yend, q, w, name_y=y_var, name_x=x_var, name_yend=yd_var, name_q=q_var, name_regimes=r_var, name_ds='columbus')
    >>> print model.name_z
    ['0.0_CONSTANT', '0.0_INC', '1.0_CONSTANT', '1.0_INC', '0.0_HOVAL', '1.0_HOVAL', 'Global_W_CRIME', 'lambda']
    >>> print model.betas
    [[ 40.40059856]
     [ -0.5550209 ]
     [ 35.69527818]
     [ -0.70983468]
     [ -0.45192502]
     [ -0.28077186]
     [  0.57378294]
     [  0.01101614]]
    >>> print np.sqrt(model.vm.diagonal())
    [ 11.2365957    1.07554818  12.56374997   0.62232066   0.51200788
       0.19175016   0.1771777 ]
    >>> print 'lambda: ', np.around(model.betas[-1], 4)
    lambda:  [ 0.011]
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
            w = REGI.w_regimes(w, regimes, self.regimes_set)
        else:
            cols2regi += [False]
        
        yend2, q2 = set_endog(y, x, w, yend, q, w_lags, lag_q)
        name_yend.append(USER.set_name_yend_sp(name_y))

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
    
