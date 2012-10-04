import numpy as np
import regimes as REGI
import user_output as USER
from utils import sphstack
from twosls import BaseTSLS
import summary_output as SUMMARY

class TSLS_Regimes(BaseTSLS, REGI.Regimes_Frame):
    """
    Two stage least squares (2SLS) with regimes

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
                   this should not contain any variables from x); cannot be
                   used in combination with h
    h            : array
                   Two dimensional array with n rows and one column for each
                   exogenous variable to use as instruments (note: this 
                   can contain variables from x); cannot be used in 
                   combination with q
    regimes      : list
                   List of n values with the mapping of each
                   observation to a regime. Assumed to be aligned with 'x'.
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


    Attributes
    ----------
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
    mean_y       : float
                   Mean of dependent variable
    std_y        : float
                   Standard deviation of dependent variable
    vm           : array
                   Variance covariance matrix (kxk)
    utu          : float
                   Sum of squared residuals
    sig2         : float
                   Sigma squared used in computations
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

    In this case we consider RD90 (resource deprivation) as an endogenous regressor.
    We tell the model that this is so by passing it in a different parameter
    from the exogenous variables (x).

    >>> yd_var = ['RD90']
    >>> yd = np.array([db.by_col(name) for name in yd_var]).T

    Because we have endogenous variables, to obtain a correct estimate of the
    model, we need to instrument for RD90. We use FP89 (families below poverty)
    for this and hence put it in the instruments parameter, 'q'.

    >>> q_var = ['FP89']
    >>> q = np.array([db.by_col(name) for name in q_var]).T

    The different regimes in this data are given according to the North and 
    South dummy (SOUTH).

    >>> r_var = 'SOUTH'
    >>> regimes = db.by_col(r_var)

    Since we want to perform tests for spatial dependence, we need to specify
    the spatial weights matrix that includes the spatial configuration of the
    observations into the error component of the model. To do that, we can open
    an already existing gal file or create a new one. In this case, we will
    create one from ``NAT.shp``.

    >>> w = pysal.rook_from_shapefile(pysal.examples.get_path("NAT.shp"))

    Unless there is a good reason not to do it, the weights have to be
    row-standardized so every row of the matrix sums to one. Among other
    things, this allows to interpret the spatial lag of a variable as the
    average value of the neighboring observations. In PySAL, this can be
    easily performed in the following way:

    >>> w.transform = 'r'

    We can now run the regression and then have a summary of the output
    by typing: model.summary
    Alternatively, we can just check the betas and standard errors of the
    parameters:

    >>> tslsr = TSLS_Regimes(y, x, yd, q, regimes, w=w, constant_regi='many', spat_diag=False, name_y=y_var, name_x=x_var, name_yend=yd_var, name_q=q_var, name_regimes=r_var, name_ds='NAT', name_w='NAT.shp')

    >>> tslsr.betas
    array([[ 3.66973562],
           [ 1.06950466],
           [ 0.14680946],
           [ 9.55873243],
           [ 1.94666348],
           [-0.30810214],
           [ 2.45864196],
           [ 3.68718119]])

    >>> tslsr.std_err
    array([ 0.46522151,  0.12074672,  0.05661795,  0.41893265,  0.16721773,
            0.06631022,  0.27538921,  0.21745974])

    """
    def __init__(self, y, x, yend, q, regimes,\
             w=None, robust=None, gwk=None, sig2n_k=True,\
             spat_diag=False, vm=False, constant_regi='many',\
             cols2regi='all', name_y=None, name_x=None,\
             name_yend=None, name_q=None, name_regimes=None,\
             name_w=None, name_gwk=None, name_ds=None, summ=True):

        n = USER.check_arrays(y, x)
        USER.check_y(y, n)
        USER.check_weights(w, y)
        USER.check_robust(robust, gwk)
        USER.check_spat_diag(spat_diag, w)
        name_yend = USER.set_name_yend(name_yend, yend)
        self.name_x_r = USER.set_name_x(name_x, x) + name_yend
        name_x = USER.set_name_x(name_x, x,constant=True)
        name_q = USER.set_name_q(name_q, q)
        self.cols2regi = cols2regi
        q, self.name_q = REGI.Regimes_Frame.__init__(self, q, \
                regimes, constant_regi=None, cols2regi='all', names=name_q)
        x, self.name_x = REGI.Regimes_Frame.__init__(self, x, \
                regimes, constant_regi, cols2regi=cols2regi, names=name_x)
        yend, self.name_yend = REGI.Regimes_Frame.__init__(self, yend, \
                regimes, constant_regi=None, \
                cols2regi=cols2regi, yend=True, names=name_yend)
        BaseTSLS.__init__(self, y=y, x=x, yend=yend, q=q, \
                robust=robust, gwk=gwk, sig2n_k=sig2n_k)
        self.constant_regi = constant_regi
        self.name_ds = USER.set_name_ds(name_ds)
        self.name_y = USER.set_name_y(name_y)
        self.name_z = self.name_x + self.name_yend
        self.name_regimes = USER.set_name_ds(name_regimes)
        self.name_h = USER.set_name_h(self.name_x, self.name_q)
        self.robust = USER.set_robust(robust)
        self.name_w = USER.set_name_w(name_w, w)
        self.name_gwk = USER.set_name_w(name_gwk, gwk)
        self.chow = REGI.Chow(self)
        if summ:
            self.title = "TWO STAGE LEAST SQUARES - REGIMES"
            SUMMARY.TSLS(reg=self, vm=vm, w=w, spat_diag=spat_diag, regimes=True)

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
    db = pysal.open(pysal.examples.get_path('NAT.dbf'),'r')
    y_var = 'HR60'
    y = np.array([db.by_col(y_var)]).T
    x_var = ['PS60','MA60','DV60','UE60']
    x = np.array([db.by_col(name) for name in x_var]).T
    yd_var = ['RD60']
    yd = np.array([db.by_col(name) for name in yd_var]).T
    q_var = ['FP89']
    q = np.array([db.by_col(name) for name in q_var]).T
    r_var = 'SOUTH'
    regimes = db.by_col(r_var)
    tslsr = TSLS_Regimes(y, x, yd, q, regimes, constant_regi='many', spat_diag=False, name_y=y_var, name_x=x_var, name_yend=yd_var, name_q=q_var, name_regimes=r_var)
    print tslsr.summary
