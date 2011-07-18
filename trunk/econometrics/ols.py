"""Ordinary Least Squares regression classes."""

__author__ = "Luc Anselin luc.anselin@asu.edu, David C. Folch david.folch@asu.edu"
import numpy as np
import copy as COPY
import numpy.linalg as la
import user_output as USER
import robust as ROBUST
from utils import RegressionProps

__all__ = ["OLS"]

class BaseOLS(RegressionProps):
    """
    Compute ordinary least squares (note: no consistency checks or
    diagnostics)

    Parameters
    ----------
    y        : array
               nx1 array of dependent variable
    x        : array
               nxj array of j independent variables
    constant : boolean
               If true it appends a vector of ones to the independent variables
               to estimate intercept (set to True by default)
    robust   : string
               If 'white' or 'hac' then a White consistent or HAC estimator of the
               variance-covariance matrix is given. If 'gls' then
               generalized least squares is performed resulting in new
               coefficient estimates along with a new variance-covariance
               matrix.
    wk       : spatial weights object
               pysal kernel weights object
               
    Attributes
    ----------

    y       : array
              nx1 array of dependent variable
    x       : array
              nxk array of independent variables (with constant added if
              constant parameter set to True)
    betas   : array
              kx1 array with estimated coefficients
    xtx     : array
              kxk array
    xtxi    : array
              kxk array of inverted xtx
    u       : array
              nx1 array of residuals
    predy   : array
              nx1 array of predicted values
    n       : int
              Number of observations
    k       : int
              Number of variables (constant included)
    utu     : float
              Sum of the squared residuals
    sig2    : float
              Sigma squared with n in the denominator
    sig2n_k : float
              Sigma squared with n-k in the denominator
    vm      : array
              Variance-covariance matrix (kxk)
    mean_y  : float
              Mean of the dependent variable
    std_y   : float
              Standard deviation of the dependent variable
              
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
    >>> ols=BaseOLS(y,X)
    >>> ols.betas
    array([[ 46.42818268],
           [  0.62898397],
           [ -0.48488854]])
    >>> ols.vm
    array([[  1.63368018e+02,  -6.12138301e+00,  -2.01939875e+00],
           [ -6.12138301e+00,   2.69616334e-01,   6.39265555e-02],
           [ -2.01939875e+00,   6.39265555e-02,   3.13263671e-02]])
    """
    def __init__(self, y, x, constant=True, robust=None, wk=None):
        if constant:
            self.x = np.hstack((np.ones(y.shape), x))
        else:
            self.x = x
        self.xtx = np.dot(self.x.T, self.x)
        self.xtxi = la.inv(self.xtx)
        xty = np.dot(self.x.T, y)
        self.betas = np.dot(self.xtxi, xty)
        predy = np.dot(self.x, self.betas)
        u = y-predy
        self.u = u
        self.predy = predy
        self.y = y
        self.n, self.k = self.x.shape

        if robust:
            self.vm = ROBUST.robust_vm(reg=self, wk=wk)

        RegressionProps()
        self._cache = {}
        self.sig2 = self.sig2n_k    #LA why? in OLS typically sig2n_k


class OLS(BaseOLS, USER.DiagnosticBuilder):
    """
    Compute ordinary least squares and return results and diagnostics.
    
    Parameters
    ----------

    y        : array
               nx1 array of dependent variable
    x        : array
               nxj array of j independent variables
    w        : spatial weights object
               if provided then spatial diagnostics are computed
    constant : boolean
               If true it appends a vector of ones to the independent variables
               to estimate intercept (set to True by default)
    name_y   : string
               Name of dependent variables for use in output
    name_x   : list of strings
               Names of independent variables for use in output
    name_ds  : string
               Name of dataset for use in output
    vm       : boolean
               If True, include variance matrix in summary results
    pred     : boolean
               If True, include y, predicted values and residuals in summary results
    

    Attributes
    ----------

    y        : array
               nx1 array of dependent variable
    x        : array
               nxk array of independent variables (with constant added if
               constant parameter set to True)
    betas    : array
               kx1 array with estimated coefficients
    u        : array
               nx1 array of residuals
    predy    : array
               nx1 array of predicted values
    n        : int
               Number of observations
    k        : int
               Number of variables (constant included)
    name_ds  : string
               dataset's name
    name_y   : string
               Dependent variable's name
    name_x   : tuple
               Independent variables' names
    mean_y   : float
               Mean value of dependent variable
    std_y    : float
               Standard deviation of dependent variable
    vm       : array
               Variance covariance matrix (kxk)
    r2       : float
               R squared
    ar2      : float
               Adjusted R squared
    utu      : float
               Sum of the squared residuals
    sig2     : float
               Sigma squared
    sig2ML   : float
               Sigma squared ML 
    f_stat   : tuple
               Statistic (float), p-value (float)
    logll    : float
               Log likelihood        
    aic      : float
               Akaike info criterion 
    schwarz  : float
               Schwarz info criterion     
    std_err  : array
               1xk array of Std.Error    
    t_stat   : list of tuples
               Each tuple contains the pair (statistic, p-value), where each is
               a float; same order as self.x
    mulColli : float
               Multicollinearity condition number
    jarque_bera : dictionary
               'jb': Jarque-Bera statistic (float); 'pvalue': p-value (float); 'df':
               degrees of freedom (int)  
    breusch_pagan : dictionary
               'bp': Breusch-Pagan statistic (float); 'pvalue': p-value (float); 'df':
               degrees of freedom (int)  
    koenker_bassett : dictionary
               'kb': Koenker-Bassett statistic (float); 'pvalue': p-value (float); 'df':
               degrees of freedom (int)  
    white    : dictionary
               'wh': White statistic (float); 'pvalue': p-value (float); 'df':
               degrees of freedom (int)  
    lm_error : tuple
               Lagrange multiplier test for spatial error model; each tuple contains
               the pair (statistic, p-value), where each is a float; only available 
               if w defined
    lm_lag   : tuple
               Lagrange multiplier test for spatial lag model; each tuple contains
               the pair (statistic, p-value), where each is a float; only available 
               if w defined
    rlm_error : tuple
               Robust lagrange multiplier test for spatial error model; each tuple 
               contains the pair (statistic, p-value), where each is a float; only 
               available if w defined
    rlm_lag   : tuple
               Robust lagrange multiplier test for spatial lag model; each tuple 
               contains the pair (statistic, p-value), where each is a float; only 
               available if w defined
    lm_sarma : tuple
               Lagrange multiplier test for spatial SARMA model; each tuple contains
               the pair (statistic, p-value), where each is a float; only available 
               if w defined
    moran_res : tuple
                Tuple containing the triple (Moran's I, standardized Moran's
                I, p-value); only available if w defined
    summary  : string
               Includes OLS regression results and diagnostics in a nice
               format for printing.        
     
    
    Examples
    --------
    >>> import numpy as np
    >>> import pysal

    Open data on Columbus neighborhood crime (49 areas) using pysal.open().
    This is the DBF associated with the Columbus shapefile.  Note that
    pysal.open() also reads data in CSV format; also, the actual OLS class
    requires data to be passed in as numpy arrays so the user can read their
    data in using any method.  

    >>> db=pysal.open("examples/columbus.dbf","r")
    
    Extract the HOVAL column (home values) from the DBF file and make it the
    dependent variable for the regression. Note that PySAL requires this to be
    an nx1 numpy array.
    
    >>> hoval = db.by_col("HOVAL")
    >>> y = np.array(hoval)
    >>> y.shape = (len(hoval), 1)

    Extract CRIME (crime) and INC (income) vectors from the DBF to be used as
    independent variables in the regression.  Note that PySAL requires this to
    be an nxj numpy array, where j is the number of independent variables (not
    including a constant). By default pysal.spreg.OLS adds a vector of ones to the
    independent variables passed in, this can be overridden by passing
    constant=False.

    >>> X = []
    >>> X.append(db.by_col("INC"))
    >>> X.append(db.by_col("CRIME"))
    >>> X = np.array(X).T

    The minimum parameters needed to run an ordinary least squares regression
    are the two numpy arrays containing the independent variable and dependent
    variables respectively.  To make the printed results more meaningful, the
    user can pass in explicit names for the variables used; this is optional.

    >>> ols = OLS(y, X, name_y='home value', name_x=['income','crime'], name_ds='columbus')

    pysal.spreg.OLS computes the regression coefficients and their standard
    errors, t-stats and p-values. It also computes a large battery of
    diagnostics on the regression. All of these results can be independently
    accessed as attributes of the regression object created by running
    pysal.spreg.OLS.  They can also be accessed at one time by printing the
    summary attribute of the regression object. In the example below, the
    parameter on crime is -0.4849, with a t-statistic of -2.6544 and p-value
    of 0.01087.

    >>> ols.betas
    array([[ 46.42818268],
           [  0.62898397],
           [ -0.48488854]])
    >>> print ols.t_stat[2][0]
    -2.73959841902
    >>> print ols.t_stat[2][1]
    0.00872490020902
    >>> ols.r2
    0.34951437785126105
    >>> print ols.summary
    REGRESSION
    ----------
    SUMMARY OF OUTPUT: ORDINARY LEAST SQUARES ESTIMATION
    ----------------------------------------------------
    Data set            :    columbus
    Dependent Variable  :  home value  Number of Observations:          49
    Mean dependent var  :     38.4362  Number of Variables   :           3
    S.D. dependent var  :     18.4661  Degrees of Freedom    :          46
    <BLANKLINE>
    R-squared           :    0.349514  F-statistic           :     12.3582
    Adjusted R-squared  :    0.321232  Prob(F-statistic)     :5.0636903e-05
    Sum squared residual:   10647.015  Log likelihood        :    -201.368
    Sigma-square        :     217.286  Akaike info criterion :     408.735
    S.E. of regression  :      14.741  Schwarz criterion     :     414.411
    Sigma-square ML     :     217.286
    S.E of regression ML:     14.7406
    <BLANKLINE>
    ----------------------------------------------------------------------------
        Variable     Coefficient       Std.Error     t-Statistic     Probability
    ----------------------------------------------------------------------------
        CONSTANT      46.4281827      12.7815499       3.6324376    0.0007042722
          income       0.6289840       0.5192459       1.2113412        0.231951
           crime      -0.4848885       0.1769926      -2.7395984       0.0087249
    ----------------------------------------------------------------------------
    <BLANKLINE>
    <BLANKLINE>
    REGRESSION DIAGNOSTICS
    MULTICOLLINEARITY CONDITION NUMBER   12.537555
    TEST ON NORMALITY OF ERRORS
    TEST                  DF          VALUE            PROB
    Jarque-Bera            2          39.706155        0.0000000
    <BLANKLINE>
    DIAGNOSTICS FOR HETEROSKEDASTICITY
    RANDOM COEFFICIENTS
    TEST                  DF          VALUE            PROB
    Breusch-Pagan test     2           5.766791        0.0559445
    Koenker-Bassett test   2           2.270038        0.3214160
    SPECIFICATION ROBUST TEST
    TEST                  DF          VALUE            PROB
    White                  5           2.906067        0.7144648
    <BLANKLINE>
    ========================= END OF REPORT ==============================


    If the optional parameters w and spat_diag are passed to pysal.spreg.OLS,
    spatial diagnostics will also be computed for the regression.  These
    include Lagrange multiplier tests and Moran's I of the residuals.  The w
    parameter is a PySAL spatial weights matrix. In this example, w is built
    directly from the shapefile columbus.shp, but w can also be read in from a
    GAL or GWT file.  In this case a rook contiguity weights matrix is built,
    but PySAL also offers queen contiguity, distance weights and k nearest
    neighbor weights among others. In the example, the Moran's I of the
    residuals is 0.2037 with a standardized value of 2.5918 and a p-value of
    0.009547.

    >>> w = pysal.weights.rook_from_shapefile("examples/columbus.shp")
    >>> ols = OLS(y, X, w, spat_diag=True, name_y='home value', name_x=['income','crime'], name_ds='columbus')
    >>> ols.betas
    array([[ 46.42818268],
           [  0.62898397],
           [ -0.48488854]])
    >>> print ols.moran_res[0]
    0.20373540938
    >>> print ols.moran_res[1]
    2.59180452208
    >>> print ols.moran_res[2]
    0.00954740031251

    """
    def __init__(self, y, x, w=None, constant=True, robust=None, wk=None,\
                        nonspat_diag=True, spat_diag=False,\
                        name_y=None, name_x=None, name_ds=None,\
                        vm=False, pred=False):
        USER.check_arrays(y, x)
        USER.check_weights(w, y)
        USER.check_robust(robust, wk)
        USER.check_spat_diag(spat_diag, w)
        BaseOLS.__init__(self, y=y, x=x, constant=constant, robust=robust, wk=wk) 
        self.title = "ORDINARY LEAST SQUARES"
        self.name_ds = USER.set_name_ds(name_ds)
        self.name_y = USER.set_name_y(name_y)
        self.name_x = USER.set_name_x(name_x, x, constant)
        self.robust = USER.set_robust(robust)
        self._get_diagnostics(w=w, beta_diag=True, nonspat_diag=nonspat_diag,\
                                    spat_diag=spat_diag, vm=vm, pred=pred)

    def _get_diagnostics(self, beta_diag=True, w=None, nonspat_diag=True,\
                              spat_diag=False, vm=False, pred=False):
        USER.DiagnosticBuilder.__init__(self, w=w, beta_diag=True,\
                                            nonspat_diag=nonspat_diag,\
                                            spat_diag=spat_diag, vm=vm,\
                                            pred=pred)

def _test():
    import doctest
    # the following line could be used to define an alternative to the '<BLANKLINE>' flag
    #doctest.BLANKLINE_MARKER = 'something better than <BLANKLINE>'
    doctest.testmod()

if __name__ == '__main__':
    _test()



