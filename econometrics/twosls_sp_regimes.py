'''
Spatial Two Stages Least Squares
'''

__author__ = "Luc Anselin luc.anselin@asu.edu, David C. Folch david.folch@asu.edu"

import copy
import numpy as np
import pysal
import numpy.linalg as la
import robust as ROBUST
import regimes as REGI
import user_output as USER
import summary_output as SUMMARY
from twosls import BaseTSLS
from utils import get_lags, set_endog, sp_att

#__all__ = ["GM_Lag"]

class GM_Lag_Regimes(BaseTSLS, REGI.Regimes_Frame):
    """
    Spatial two stage least squares (S2SLS) with regimes; 
    Anselin (1988) [1]_

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
                   this should not contain any variables from x); cannot be
                   used in combination with h
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
    w            : pysal W object
                   Spatial weights object 
    w_lags       : integer
                   Orders of W to include as instruments for the spatially
                   lagged dependent variable. For example, w_lags=1, then
                   instruments are WX; if w_lags=2, then WX, WWX; and so on.
    lag_q        : boolean
                   If True, then include spatial lags of the additional 
                   instruments (q).
    robust       : string
                   If 'white', then a White consistent estimator of the
                   variance-covariance matrix is given.  If 'hac', then a
                   HAC consistent estimator of the variance-covariance
                   matrix is given. Default set to None. 
    gwk          : pysal W object
                   Kernel spatial weights needed for HAC estimation. Note:
                   matrix must have ones along the main diagonal.
    sig2n_k      : boolean
                   If True, then use n-k to estimate sigma^2. If False, use n.
    spat_diag    : boolean
                   If True, then compute Anselin-Kelejian test
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
    name_gwk     : string
                   Name of kernel weights matrix for use in output
    name_ds      : string
                   Name of dataset for use in output

    Attributes
    ----------
    summary      : string
                   Summary of regression results and diagnostics (note: use in
                   conjunction with the print command)
    betas        : array
                   kx1 array of estimated coefficients
    u            : array
                   nx1 array of residuals
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
    kstar        : integer
                   Number of endogenous variables. 
    y            : array
                   nx1 array for dependent variable
    x            : array
                   Two dimensional array with n rows and one column for each
                   independent (exogenous) variable, including the constant
    yend         : array
                   Two dimensional array with n rows and one column for each
                   endogenous variable
    q            : array
                   Two dimensional array with n rows and one column for each
                   external exogenous variable used as instruments 
    z            : array
                   nxk array of variables (combination of x and yend)
    h            : array
                   nxl array of instruments (combination of x and q)
    robust       : string
                   Adjustment for robust standard errors
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
    utu          : float
                   Sum of squared residuals
    sig2         : float
                   Sigma squared used in computations
    std_err      : array
                   1xk array of standard errors of the betas    
    z_stat       : list of tuples
                   z statistic; each tuple contains the pair (statistic,
                   p-value), where each is a float
    ak_test      : tuple
                   Anselin-Kelejian test; tuple contains the pair (statistic,
                   p-value)
    name_y       : string
                   Name of dependent variable for use in output
    name_x       : list of strings
                   Names of independent variables for use in output
    name_yend    : list of strings
                   Names of endogenous variables for use in output
    name_z       : list of strings
                   Names of exogenous and endogenous variables for use in 
                   output
    name_q       : list of strings
                   Names of external instruments
    name_h       : list of strings
                   Names of all instruments used in ouput
    name_w       : string
                   Name of weights matrix for use in output
    name_gwk     : string
                   Name of kernel weights matrix for use in output
    name_ds      : string
                   Name of dataset for use in output
    title        : string
                   Name of the regression method used
    sig2n        : float
                   Sigma squared (computed with n in the denominator)
    sig2n_k      : float
                   Sigma squared (computed with n-k in the denominator)
    hth          : float
                   H'H
    hthi         : float
                   (H'H)^-1
    varb         : array
                   (Z'H (H'H)^-1 H'Z)^-1
    zthhthi      : array
                   Z'H(H'H)^-1
    pfora1a2     : array
                   n(zthhthi)'varb
    regimes      : list
                   List of n values with the mapping of each
                   observation to a regime. Assumed to be aligned with 'x'.
    constant_regi: [False, 'one', 'many']
                   Ignored if regimes=False. Constant option for regimes.
                   Switcher controlling the constant term setup. It may take
                   the following values:
                    
                     *  False: no constant term is appended in any way
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

    .. [1] Anselin, L. (1988) "Spatial Econometrics: Methods and Models".
    Kluwer Academic Publishers. Dordrecht.

    
    Examples
    --------

    """
    def __init__(self, y, x, regimes, yend=None, q=None,\
                 w=None, w_lags=1, lag_q=True,\
                 robust=None, gwk=None, sig2n_k=False,\
                 spat_diag=False, constant_regi='many',\
                 cols2regi='all',\
                 vm=False, name_y=None, name_x=None,\
                 name_yend=None, name_q=None, name_regimes=None,\
                 name_w=None, name_gwk=None, name_ds=None):

        n = USER.check_arrays(y, x)
        USER.check_y(y, n)
        USER.check_weights(w, y)
        USER.check_robust(robust, gwk)
        USER.check_spat_diag(spat_diag, w)
        name_yend = USER.set_name_yend(name_yend, yend)
        self.name_x_r = USER.set_name_x(name_x, x) + name_yend
        name_q = USER.set_name_q(name_q, q)
        self.cols2regi = cols2regi
        print 'y2'
        y2, name_y2 = REGI.Regimes_Frame.__init__(self, y, \
                ['Null'], regimes, constant_regi=None, \
                cols2regi=cols2regi)
        print 'x'
        x, name_x = REGI.Regimes_Frame.__init__(self, x, name_x, \
                regimes, constant_regi, cols2regi=cols2regi)
        if q:
            print 'q'
            q, name_q = REGI.Regimes_Frame.__init__(self, q, name_q, \
                regimes, constant_regi=None, cols2regi='all')
        if yend:
            print 'yend'
            yend, name_yend = REGI.Regimes_Frame.__init__(self, yend, \
                name_yend, regimes, constant_regi=None, \
                cols2regi=cols2regi, yend=True)
        yend2, q2 = set_endog(y2, x, w, yend, q, w_lags, lag_q)

        BaseTSLS.__init__(self, y=y, x=x, yend=yend2, q=q2, \
                robust=robust, gwk=gwk, sig2n_k=sig2n_k)
        self.name_x_r.append(USER.set_name_yend_sp(name_y))        
        name_q.extend(USER.set_name_q_sp(name_x, w_lags, name_q, lag_q, force_all=True))
        print self.betas

        #if cols2regi=='all':
        #    cols2regi = [True] * (x.shape[1]+yend2.shape[1]-1)
        #    cols2regi.extend([False])
        #TSLS_Regimes.__init__(self, y=y, x=x, yend=yend2, q=q2,\
        #     regimes=regimes, w=w, robust=robust, gwk=gwk,\
        #     sig2n_k=sig2n_k, spat_diag=spat_diag, vm=vm,\
        #     constant_regi=constant_regi, cols2regi=cols2regi, name_y=name_y,\
        #     name_x=name_x, name_yend=name_yend, name_q=name_q,\
        #     name_regimes=name_regimes, name_w=name_w, name_gwk=name_gwk,\
        #     name_ds=name_ds)
        self.name_ds = USER.set_name_ds(name_ds)
        self.name_y = USER.set_name_y(name_y)
        self.name_x = USER.set_name_x(name_x, x, regi=True)
        self.name_yend = USER.set_name_yend(name_yend, yend)
        self.name_z = self.name_x + self.name_yend
        self.name_q = USER.set_name_q(name_q, q)
        self.name_regimes = USER.set_name_ds(name_regimes)
        self.name_h = USER.set_name_h(self.name_x, self.name_q)
        self.robust = USER.set_robust(robust)
        self.name_w = USER.set_name_w(name_w, w)
        self.name_gwk = USER.set_name_w(name_gwk, gwk)
        self.predy_e, self.e_pred = sp_att(w,self.y,self.predy,\
                      yend2[:,-1].reshape(self.n,1),self.betas[-1])
        self.title = "SPATIAL TWO STAGE LEAST SQUARES - REGIMES"        
        SUMMARY.GM_Lag(reg=self, w=w, vm=vm, spat_diag=spat_diag, regimes=True)


def _test():
    import doctest
    start_suppress = np.get_printoptions()['suppress']
    np.set_printoptions(suppress=True)    
    doctest.testmod()
    np.set_printoptions(suppress=start_suppress)


if __name__ == '__main__':
    #_test()        
    
    import numpy as np
    import pysal
    """
    db = pysal.open(pysal.examples.get_path("columbus.dbf"),'r')
    y_var = 'CRIME'
    y = np.array([db.by_col(y_var)]).reshape(49,1)
    #"""
    """
    x_var = ['INC','HOVAL']
    x = np.array([db.by_col(name) for name in x_var]).T    
    yd, yd_var = None, None
    q, q_var = None, None
    """
    """
    x_var = ['INC']
    x = np.array([db.by_col(name) for name in x_var]).T
    yd_var = ['HOVAL']
    yd = np.array([db.by_col(name) for name in yd_var]).T
    q_var = ['DISCBD']
    q = np.array([db.by_col(name) for name in q_var]).T
    #"""
    """
    r_var = 'NSA'
    regimes = db.by_col(r_var)
    w = pysal.rook_from_shapefile(pysal.examples.get_path("columbus.shp"))
    w.transform = 'r'
    model = GM_Lag_Regimes(y, x, regimes, yend=yd, q=q, w=w, constant_regi='many', spat_diag=True, name_y=y_var, name_x=x_var, name_yend=yd_var, name_q=q_var, name_regimes=r_var, name_ds='columbus', name_w='columbus.gal')
    print model.summary


    """
    db = pysal.open(pysal.examples.get_path('NAT.dbf'),'r')
    y_var = 'HR60'
    y = np.array([db.by_col(y_var)]).T
    x_var = ['PS60','MA60','DV60','UE60']
    x = np.array([db.by_col(name) for name in x_var]).T
    yd, yd_var = None, None
    q, q_var = None, None
    r_var = 'SOUTH'
    regimes = db.by_col(r_var)
    w = pysal.queen_from_shapefile(pysal.examples.get_path("NAT.shp"))
    w.transform = 'r'
    model = GM_Lag_Regimes(y, x, regimes, yend=yd, q=q, w=w, constant_regi='many', spat_diag=False, name_y=y_var, name_x=x_var, name_yend=yd_var, name_q=q_var, name_regimes=r_var, name_ds='columbus', name_w='columbus.gal')
    print model.summary




