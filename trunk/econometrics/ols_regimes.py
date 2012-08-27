"""This is a place holder that has all the unique stuff from ols.py and
test_ols.py related to regimes."""


#'''
#commenting out entire code so the testing module doesn't find it :)

import regimes as REGI
import user_output as USER
from ols import BaseOLS
import summary_output as SUMMARY


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

    Open data on Columbus neighborhood crime (49 areas) using pysal.open().
    This is the DBF associated with the Columbus shapefile.  Note that
    pysal.open() also reads data in CSV format; also, the actual OLS class
    requires data to be passed in as numpy arrays so the user can read their
    data in using any method.  

    >>> db = pysal.open(pysal.examples.get_path('columbus.dbf'),'r')
    
    Extract the HOVAL column (home values) from the DBF file and make it the
    dependent variable for the regression. Note that PySAL requires this to be
    an nx1 numpy array.
    
    >>> hoval = db.by_col("HOVAL")
    >>> y = np.array(hoval)
    >>> y.shape = (len(hoval), 1)

    Extract CRIME (crime) and INC (income) vectors from the DBF to be used as
    independent variables in the regression.  Note that PySAL requires this to
    be an nxj numpy array, where j is the number of independent variables (not
    including a constant). pysal.spreg.OLS adds a vector of ones to the
    independent variables passed in.

    >>> X = []
    >>> X.append(db.by_col("INC"))
    >>> X.append(db.by_col("CRIME"))
    >>> X = np.array(X).T

    >>> regimes = db.by_col("NSA")
    >>> olsr = OLS_Regimes(y, X, regimes, nonspat_diag=False)
    >>> olsr.betas
    array([[  9.386956  ],
           [  1.86338981],
           [ -0.09091956],
           [ 57.42063184],
           [  0.6347145 ],
           [ -0.66190849]])
    >>> olsr.std_err
    array([ 21.29552114,   0.79528657,   0.28460755,  16.79327777,
             0.85352708,   0.22389296])
    >>> olsr.cols2regi
    [True, True, True]
    """
    def __init__(self, y, x, regimes,\
                 w=None, robust=None, gwk=None, sig2n_k=True,\
                 nonspat_diag=True, spat_diag=False, moran=False,\
                 vm=False, constant_regi='many',
                 cols2regi='all', name_y=None, name_x=None,\
                 name_w=None, name_gwk=None, name_ds=None):

        n = USER.check_arrays(y, x)
        USER.check_y(y, n)
        USER.check_weights(w, y)
        USER.check_robust(robust, gwk)
        USER.check_spat_diag(spat_diag, w)
        x, name_x = REGI.Regimes_Frame.__init__(self, x, name_x, \
                regimes, constant_regi, cols2regi)
        BaseOLS.__init__(self, y=y, x=x, robust=robust, gwk=gwk, \
                sig2n_k=sig2n_k)
        self.title = "ORDINARY LEAST SQUARES - REGIMES"
        self.name_ds = USER.set_name_ds(name_ds)
        self.name_y = USER.set_name_y(name_y)
        self.name_x = USER.set_name_x(name_x, x, regi=True)
        self.robust = USER.set_robust(robust)
        self.name_w = USER.set_name_w(name_w, w)
        self.name_gwk = USER.set_name_w(name_gwk, gwk)
        SUMMARY.OLS(reg=self, vm=vm, w=w, nonspat_diag=nonspat_diag,\
                    spat_diag=spat_diag, moran=moran)




if __name__ == '__main__':
    _test()
    import numpy as np
    import pysal
    db = pysal.open(pysal.examples.get_path('columbus.dbf'),'r')
    hoval = db.by_col("HOVAL")
    y = np.array(hoval)
    y.shape = (len(hoval), 1)
    X = []
    X.append(db.by_col("INC"))
    X.append(db.by_col("CRIME"))
    X = np.array(X).T
    regimes = db.by_col("NSA")
    brk = int(y.shape[0]/2)
    regimes = [1] * y.shape[0]; regimes[:brk] = [0]*brk





######################################
#code from test_ols.py


    def test_OLS_Regimes(self):
        regimes = [1] * self.w.n
        regimes[:int(self.w.n/2.)] = [0] * int(self.w.n/2.)
        ols = EC.ols.OLS_Regimes(self.y, self.X, regimes, w=self.w, spat_diag=True, moran=True, \
                name_y='home value', name_x=['income','crime'], \
                name_ds='columbus', nonspat_diag=False)
        
        #np.testing.assert_array_almost_equal(ols.aic, \
        #        408.73548964604873 ,7)
        #np.testing.assert_array_almost_equal(ols.ar2, \
        #        0.32123239427957662 ,7)
        np.testing.assert_array_almost_equal(ols.betas, \
                np.array([[ 62.00971868, \
                           0.54875832, \
                          -0.73304867, \
                          13.06125529, \
                           1.5544147 , \
                          -0.08041612]]).T, 7)
        #bp = np.array([ 2, 5.7667905131212587, 0.05594449410070558])
        #ols_bp = np.array([ols.breusch_pagan['df'], ols.breusch_pagan['bp'], ols.breusch_pagan['pvalue']])
        #np.testing.assert_array_almost_equal(bp, ols_bp, 7)
        #np.testing.assert_array_almost_equal(ols.f_stat, \
        #    (12.358198885356581, 5.0636903313953024e-05), 7)
        #jb = np.array([2, 39.706155069114878, 2.387360356860208e-09])
        #ols_jb = np.array([ols.jarque_bera['df'], ols.jarque_bera['jb'], ols.jarque_bera['pvalue']])
        #np.testing.assert_array_almost_equal(ols_jb,jb, 7)
        np.testing.assert_equal(ols.k,  6)
        kb = {'df': 2, 'kb': 2.2700383871478675, 'pvalue': 0.32141595215434604}
        #for key in kb:
        #    self.assertAlmostEqual(ols.koenker_bassett[key],  kb[key], 7)
        np.testing.assert_array_almost_equal(ols.lm_error, \
            (0.48910653,  0.48432613),7)
        np.testing.assert_array_almost_equal(ols.lm_lag, \
            (0.04972876,  0.82353592), 7)
        np.testing.assert_array_almost_equal(ols.lm_sarma, \
                (0.54831396,  0.76021273), 7)
        #np.testing.assert_array_almost_equal(ols.logll, \
        #        -201.3677448230244 ,7)
        np.testing.assert_array_almost_equal(ols.mean_y, \
            38.436224469387746,7)
        np.testing.assert_array_almost_equal(ols.moran_res[0], \
            0.06993615168844197,7)
        np.testing.assert_array_almost_equal(ols.moran_res[1], \
            1.5221890693566746,7)
        np.testing.assert_array_almost_equal(ols.moran_res[2], \
            0.12796171317640753,7)
        #np.testing.assert_array_almost_equal(ols.mulColli, \
        #    12.537554873824675 ,7)
        np.testing.assert_equal(ols.n,  49)
        np.testing.assert_equal(ols.name_ds,  'columbus')
        np.testing.assert_equal(ols.name_gwk,  None)
        np.testing.assert_equal(ols.name_w,  'unknown')
        np.testing.assert_equal(ols.name_x,  ['0_-_CONSTANT', '0_-_income',
            '0_-_crime', '1_-_CONSTANT', '1_-_income', '1_-_crime'])
        np.testing.assert_equal(ols.name_y,  'home value')
        np.testing.assert_array_almost_equal(ols.predy[3], np.array([
            40.72470539]),7)
        np.testing.assert_array_almost_equal(ols.r2, \
                0.4880852959523868 ,7)
        np.testing.assert_array_almost_equal(ols.rlm_error, \
                (0.4985852 ,  0.48012244),7)
        np.testing.assert_array_almost_equal(ols.rlm_lag, \
            (0.05920743,  0.80775305), 7)
        np.testing.assert_equal(ols.robust,  'unadjusted')
        #np.testing.assert_array_almost_equal(ols.schwarz, \
        #    414.41095054038061,7 )
        np.testing.assert_array_almost_equal(ols.sig2, \
            194.858482437219,7 )
        #np.testing.assert_array_almost_equal(ols.sig2ML, \
        #    217.28602192257551,7 )
        np.testing.assert_array_almost_equal(ols.sig2n, \
                170.9982600979677, 7)
 
        np.testing.assert_array_almost_equal(ols.t_stat[2][0], \
                -3.2474266949819044,7)
        np.testing.assert_array_almost_equal(ols.t_stat[2][1], \
                0.0022611435827165284,7)


#'''
