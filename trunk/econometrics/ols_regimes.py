"""
Ordinary Least Squares regression with regimes.
"""
 
__author__ = "Luc Anselin luc.anselin@asu.edu, Pedro V. Amaral pedro.amaral@asu.edu, Daniel Arribas-Bel darribas@asu.edu"

import regimes as REGI
import user_output as USER
import multiprocessing as mp
from ols import BaseOLS
from utils import set_warn
from robust import hac_multi
import summary_output as SUMMARY
import numpy as np
from platform import system


class OLS_Regimes(BaseOLS, REGI.Regimes_Frame):
    """
    Ordinary least squares with results and diagnostics.
    
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
                   Spatial weights object (required if running spatial
                   diagnostics)
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
    nonspat_diag : boolean
                   If True, then compute non-spatial diagnostics on
                   the regression.
    spat_diag    : boolean
                   If True, then compute Lagrange multiplier tests (requires
                   w). Note: see moran for further tests.
    moran        : boolean
                   If True, compute Moran's I on the residuals. Note:
                   requires spat_diag=True.
    white_test   : boolean
                   If True, compute White's specification robust test.
                   (requires nonspat_diag=True)
    vm           : boolean
                   If True, include variance-covariance matrix in summary
                   results
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
    regime_err_sep  : boolean
                   If True, a separate regression is run for each regime.
    name_y       : string
                   Name of dependent variable for use in output
    name_x       : list of strings
                   Names of independent variables for use in output
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
    robust       : string
                   Adjustment for robust standard errors
    mean_y       : float
                   Mean of dependent variable
    std_y        : float
                   Standard deviation of dependent variable
    vm           : array
                   Variance covariance matrix (kxk)
    r2           : float
                   R squared
    ar2          : float
                   Adjusted R squared
    utu          : float
                   Sum of squared residuals
    sig2         : float
                   Sigma squared used in computations
    sig2ML       : float
                   Sigma squared (maximum likelihood)
    f_stat       : tuple
                   Statistic (float), p-value (float)
    logll        : float
                   Log likelihood
    aic          : float
                   Akaike information criterion 
    schwarz      : float
                   Schwarz information criterion     
    std_err      : array
                   1xk array of standard errors of the betas    
    t_stat       : list of tuples
                   t statistic; each tuple contains the pair (statistic,
                   p-value), where each is a float
    mulColli     : float
                   Multicollinearity condition number
    jarque_bera  : dictionary
                   'jb': Jarque-Bera statistic (float); 'pvalue': p-value
                   (float); 'df': degrees of freedom (int)  
    breusch_pagan : dictionary
                    'bp': Breusch-Pagan statistic (float); 'pvalue': p-value
                    (float); 'df': degrees of freedom (int)  
    koenker_bassett : dictionary
                      'kb': Koenker-Bassett statistic (float); 'pvalue':
                      p-value (float); 'df': degrees of freedom (int)  
    white         : dictionary
                    'wh': White statistic (float); 'pvalue': p-value (float);
                    'df': degrees of freedom (int)  
    lm_error      : tuple
                    Lagrange multiplier test for spatial error model; tuple
                    contains the pair (statistic, p-value), where each is a
                    float 
    lm_lag        : tuple
                    Lagrange multiplier test for spatial lag model; tuple
                    contains the pair (statistic, p-value), where each is a
                    float 
    rlm_error     : tuple
                    Robust lagrange multiplier test for spatial error model;
                    tuple contains the pair (statistic, p-value), where each
                    is a float
    rlm_lag       : tuple
                    Robust lagrange multiplier test for spatial lag model;
                    tuple contains the pair (statistic, p-value), where each
                    is a float
    lm_sarma      : tuple
                    Lagrange multiplier test for spatial SARMA model; tuple
                    contains the pair (statistic, p-value), where each is a
                    float
    moran_res     : tuple
                    Moran's I for the residuals; tuple containing the triple
                    (Moran's I, standardized Moran's I, p-value)
    name_y        : string
                    Name of dependent variable for use in output
    name_x        : list of strings
                    Names of independent variables for use in output
    name_w        : string
                    Name of weights matrix for use in output
    name_gwk      : string
                    Name of kernel weights matrix for use in output
    name_ds       : string
                    Name of dataset for use in output
    title         : string
                    Name of the regression method used
    sig2n        : float
                   Sigma squared (computed with n in the denominator)
    sig2n_k      : float
                   Sigma squared (computed with n-k in the denominator)
    xtx          : float
                   X'X
    xtxi         : float
                   (X'X)^-1
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
    regime_err_sep  : boolean
                   If True, a separate regression is run for each regime.
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

    
    Examples
    --------
    >>> import numpy as np
    >>> import pysal

    Open data on NCOVR US County Homicides (3085 areas) using pysal.open().
    This is the DBF associated with the NAT shapefile.  Note that
    pysal.open() also reads data in CSV format; since the actual class
    requires data to be passed in as numpy arrays, the user can read their
    data in using any method.  

    >>> db = pysal.open(pysal.examples.get_path("NAT.dbf"),'r')
 
    Extract the HR90 column (homicide rates in 1990) from the DBF file and make it
    the dependent variable for the regression. Note that PySAL requires this to be
    an numpy array of shape (n, 1) as opposed to the also common shape of (n, )
    that other packages accept.
    
    >>> y_var = 'HR90'
    >>> y = db.by_col(y_var)
    >>> y = np.array(y).reshape(len(y), 1)

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

    >>> olsr = OLS_Regimes(y, x, regimes, nonspat_diag=False, name_y=y_var, name_x=['APS90','UE90'], name_regimes=r_var, name_ds='NAT')
    >>> olsr.betas
    array([[ 0.39642899],
           [ 0.65583299],
           [ 0.48703937],
           [ 5.59835   ],
           [ 1.16210453],
           [ 0.53163886]])
    >>> olsr.std_err
    array([ 0.31880436,  0.12413205,  0.04661535,  0.38716735,  0.17888871,
            0.04908804])
    >>> olsr.cols2regi
    'all'
    >>> print olsr.summary
    REGRESSION
    ----------
    SUMMARY OF OUTPUT: ORDINARY LEAST SQUARES - REGIMES
    ---------------------------------------------------
    Data set            :         NAT
    Dependent Variable  :        HR90               Number of Observations:        3085
    Mean dependent var  :      6.1829               Number of Variables   :           6
    S.D. dependent var  :      6.6414               Degrees of Freedom    :        3079
    R-squared           :      0.2851
    Adjusted R-squared  :      0.2839
    <BLANKLINE>
    ------------------------------------------------------------------------------------
                Variable     Coefficient       Std.Error     t-Statistic     Probability
    ------------------------------------------------------------------------------------
              0_CONSTANT       0.3964290       0.3188044       1.2434867       0.2137832
                 0_APS90       0.6558330       0.1241320       5.2833494       0.0000001
                  0_UE90       0.4870394       0.0466153      10.4480471       0.0000000
              1_CONSTANT       5.5983500       0.3871674      14.4597677       0.0000000
                 1_APS90       1.1621045       0.1788887       6.4962431       0.0000000
                  1_UE90       0.5316389       0.0490880      10.8303144       0.0000000
    ------------------------------------------------------------------------------------
    Regimes variable: SOUTH
    <BLANKLINE>
    REGIMES DIAGNOSTICS - CHOW TEST
                     VARIABLE        DF        VALUE           PROB
                     CONSTANT         1      107.579486        0.0000000
                        APS90         1        5.406269        0.0200646
                         UE90         1        0.434056        0.5100055
                  Global test         3      719.076563        0.0000000
    ================================ END OF REPORT =====================================
    """
    def __init__(self, y, x, regimes,\
                 w=None, robust=None, gwk=None, sig2n_k=True,\
                 nonspat_diag=True, spat_diag=False, moran=False, white_test=False,\
                 vm=False, constant_regi='many', cols2regi='all',\
                 regime_err_sep=False, cores=None,\
                 name_y=None, name_x=None, name_regimes=None,\
                 name_w=None, name_gwk=None, name_ds=None):         
        
        n = USER.check_arrays(y, x)
        USER.check_y(y, n)
        USER.check_weights(w, y)
        USER.check_robust(robust, gwk)
        USER.check_spat_diag(spat_diag, w)
        self.name_x_r = USER.set_name_x(name_x, x)
        self.constant_regi = constant_regi
        self.cols2regi = cols2regi        
        self.name_w = USER.set_name_w(name_w, w)
        self.name_gwk = USER.set_name_w(name_gwk, gwk)
        self.name_ds = USER.set_name_ds(name_ds)
        self.name_y = USER.set_name_y(name_y)
        self.name_regimes = USER.set_name_ds(name_regimes)
        self.n = n        
        if regime_err_sep == True:
            name_x = USER.set_name_x(name_x, x)
            self.y = y
            if cols2regi == 'all':
                cols2regi = [True] * (x.shape[1])
            self.regimes_set = REGI._get_regimes_set(regimes)
            if w:
                w_i,regi_ids,warn = REGI.w_regimes(w, regimes, self.regimes_set, transform=True, get_ids=True, min_n=len(self.cols2regi)+1)
                set_warn(self,warn)
            else:
                regi_ids = dict((r, list(np.where(np.array(regimes) == r)[0])) for r in self.regimes_set)
                w_i = None
            if set(cols2regi) == set([True]):
                self._ols_regimes_multi(x, w_i, regi_ids, cores,\
                 gwk, sig2n_k, robust, nonspat_diag, spat_diag, vm, name_x, moran, white_test)
            else:
                raise Exception, "All coefficients must vary accross regimes if regime_err_sep = True."
        else:
            name_x = USER.set_name_x(name_x, x,constant=True)
            x, self.name_x = REGI.Regimes_Frame.__init__(self, x,\
                    regimes, constant_regi, cols2regi, name_x)
            BaseOLS.__init__(self, y=y, x=x, robust=robust, gwk=gwk, \
                    sig2n_k=sig2n_k)
            self.title = "ORDINARY LEAST SQUARES - REGIMES"
            self.robust = USER.set_robust(robust)
            self.chow = REGI.Chow(self)
            SUMMARY.OLS(reg=self, vm=vm, w=w, nonspat_diag=nonspat_diag,\
                        spat_diag=spat_diag, moran=moran, white_test=white_test, regimes=True)

    def _ols_regimes_multi(self, x, w_i, regi_ids, cores,\
                 gwk, sig2n_k, robust, nonspat_diag, spat_diag, vm, name_x, moran, white_test):
        pool = mp.Pool(cores)
        results_p = {}
        for r in self.regimes_set:
            if system() == 'Windows':
                is_win = True
                results_p[r] = _work(*(self.y,x,regi_ids,r,robust,sig2n_k,self.name_ds,self.name_y,name_x,self.name_w,self.name_regimes))
            else:
                results_p[r] = pool.apply_async(_work,args=(self.y,x,regi_ids,r,robust,sig2n_k,self.name_ds,self.name_y,name_x,self.name_w,self.name_regimes))
                is_win = False
        self.kryd = 0
        self.kr = x.shape[1]+1
        self.kf = 0
        self.nr = len(self.regimes_set)
        self.vm = np.zeros((self.nr*self.kr,self.nr*self.kr),float)
        self.betas = np.zeros((self.nr*self.kr,1),float)
        self.u = np.zeros((self.n,1),float)
        self.predy = np.zeros((self.n,1),float)
        if not is_win:
            pool.close()
            pool.join()
        results = {}
        self.name_y, self.name_x = [],[]
        counter = 0
        for r in self.regimes_set:
            if is_win:
                results[r] = results_p[r]
            else:
                results[r] = results_p[r].get()
            if w_i:
                results[r].w = w_i[r]
            else:
                results[r].w = None
            self.vm[(counter*self.kr):((counter+1)*self.kr),(counter*self.kr):((counter+1)*self.kr)] = results[r].vm
            self.betas[(counter*self.kr):((counter+1)*self.kr),] = results[r].betas
            self.u[regi_ids[r],]=results[r].u
            self.predy[regi_ids[r],]=results[r].predy
            self.name_y += results[r].name_y
            self.name_x += results[r].name_x
            counter += 1
        self.multi = results
        self.hac_var = x
        if robust == 'hac':
            hac_multi(self,gwk)
        self.chow = REGI.Chow(self)            
        SUMMARY.OLS_multi(reg=self, multireg=self.multi, vm=vm, nonspat_diag=nonspat_diag, spat_diag=spat_diag, moran=moran, white_test=white_test, regimes=True)

def _work(y,x,regi_ids,r,robust,sig2n_k,name_ds,name_y,name_x,name_w,name_regimes):
    y_r = y[regi_ids[r]]
    x_r = x[regi_ids[r]]
    x_constant = USER.check_constant(x_r)
    if robust == 'hac':
        robust = None
    model = BaseOLS(y_r, x_constant, robust=robust, sig2n_k=sig2n_k)
    model.title = "ORDINARY LEAST SQUARES ESTIMATION - REGIME %s" %r
    model.robust = USER.set_robust(robust)
    model.name_ds = name_ds
    model.name_y = '%s_%s'%(str(r), name_y)
    model.name_x = ['%s_%s'%(str(r), i) for i in name_x]
    model.name_w = name_w
    model.name_regimes = name_regimes
    return model
            
def _test():
    import doctest
    start_suppress = np.get_printoptions()['suppress']
    np.set_printoptions(suppress=True)    
    doctest.testmod()
    np.set_printoptions(suppress=start_suppress)

if __name__ == '__main__':
    _test()
    import numpy as np
    import pysal
    db = pysal.open(pysal.examples.get_path('columbus.dbf'),'r')
    y_var = 'CRIME'
    y = np.array([db.by_col(y_var)]).reshape(49,1)
    x_var = ['INC','HOVAL']
    x = np.array([db.by_col(name) for name in x_var]).T
    r_var = 'NSA'
    regimes = db.by_col(r_var)
    w = pysal.rook_from_shapefile(pysal.examples.get_path("columbus.shp"))
    w.transform = 'r'
    olsr = OLS_Regimes(y, x, regimes, w=w, constant_regi='many', nonspat_diag=False, spat_diag=True, name_y=y_var, name_x=['AINC','HOVAL'], name_ds='columbus', name_regimes=r_var, name_w='columbus.gal')
    print olsr.summary

