import numpy as np
import numpy.linalg as la
import user_output as USER

class Regression_Props:
    """
    Helper class that adds common regression properties to any regression
    class that inherits it.  It takes no parameters.  See OLS_dev for example
    usage.

    Parameters
    ----------

    Attributes
    ----------
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
              
    """

    @property
    def utu(self):
        if 'utu' not in self._cache:
            self._cache['utu'] = np.sum(self.u**2)
        return self._cache['utu']
    @property
    def sig2n(self):
        if 'sig2n' not in self._cache:
            self._cache['sig2n'] = self.utu / self.n
        return self._cache['sig2n']
    @property
    def sig2n_k(self):
        if 'sig2n_k' not in self._cache:
            self._cache['sig2n_k'] = self.utu / (self.n-self.k)
        return self._cache['sig2n_k']
    @property
    def m(self):
        if 'm' not in self._cache:
            xtxixt = np.dot(self.xtxi,self.xt)
            xxtxixt = np.dot(self.x, xtxixt)
            self._cache['m'] = np.eye(self.n) - xxtxixt
        return self._cache['m']
    @property
    def vm(self):
        if 'vm' not in self._cache:
            self._cache['vm'] = np.dot(self.sig2n_k, self.xtxi)
        return self._cache['vm']
    
    @property
    def mean_y(self):
        if 'mean_y' not in self._cache:
            self._cache['mean_y']=np.mean(self.y)
        return self._cache['mean_y']
    @property
    def std_y(self):
        if 'std_y' not in self._cache:
            self._cache['std_y']=np.std(self.y)
        return self._cache['std_y']
    


class OLS_dev(Regression_Props):
    """
    OLS class to do all the computations

    NOTE: no consistency checks
    
    Maximal Complexity: O(n^3)

    Parameters
    ----------
    x        : array
               nxk array of independent variables (assumed to be aligned with y)
    y        : array
               nx1 array of dependent variable
    constant : boolean
               If true it appends a vector of ones to the independent variables
               to estimate intercept (set to True by default)

    Attributes
    ----------

    x       : array
              nxk array of independent variables (assumed to be aligned with y)
    y       : array
              nx1 array of dependent variable
    betas   : array
              kx1 array with estimated coefficients
    xt      : array
              kxn array of transposed independent variables
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
              Number of variables
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
    >>> y = np.array(db.by_col("CRIME"))
    >>> y = np.reshape(y, (49,1))
    >>> X = []
    >>> X.append(db.by_col("INC"))
    >>> X.append(db.by_col("HOVAL"))
    >>> X = np.array(X).T
    >>> ols=OLS_dev(X,y)
    >>> ols.betas
    array([[ 68.6189611 ],
           [ -1.59731083],
           [ -0.27393148]])
    
    """
    def __init__(self,x,y,constant=True):
        if constant:
            x = np.hstack((np.ones(y.shape),x))
        self.set_x(x)
        xty = np.dot(x.T,y)
        self.betas = np.dot(self.xtxi,xty)
        predy = np.dot(x,self.betas)
        u = y-predy
        self.u = u
        self.predy = predy
        self.y = y
        self.n, self.k = x.shape
        Regression_Props()
        self._cache = {}
        self.sig2 = self.sig2n_k

    def set_x(self, x):
        self.x = x
        self.xt = x.T
        self.xtx = np.dot(self.x.T,self.x)
        self.xtxi = la.inv(self.xtx)

class OLS(OLS_dev, USER.Diagnostic_Builder):
    """
    OLS class for end-user (gives back only results and diagnostics)
    
    Maximal Complexity: O(n^3)

    Parameters
    ----------

    x        : array
               nxk array of independent variables (assumed to be aligned with y)

    y        : array
               nx1 array of dependent variable

    names    : tuple
               used in summary output, the sequence is (dataset name, dependent name, independent names)
    
    constant : boolean
               If true it appends a vector of ones to the independent variables
               to estimate intercept (set to True by default)

    vm       : boolean
               if True, include variance matrix in summary results

    pred     : boolean
               if True, include y, predicted values and residuals in summary results

    
    Attributes
    ----------

    x        : array
               nxk array of independent variables (assumed to be aligned with y)
    y        : array
               nx1 array of dependent variable
    betas    : array
               kx1 array with estimated coefficients
    u        : array
               nx1 array of residuals
    predy    : array
               nx1 array of predicted values
    n        : int
               Number of observations
    k        : int
               Number of variables
    name_ds  : string
               dataset's name
    name_y   : string
               dependent variable's name
    name_x   : tuple
               independent variables' names
    mean_y   : float
               mean value of dependent variable
    std_y    : float
               standard deviation of dependent variable
    vm       : array
               variance covariance matrix (kxk)
    r2       : float
               R square
    ar2      : float
               adjusted R square
    utu      : float
               Sum of the squared residuals
    sig2     : float
               sigma squared
    sig2ML   : float
               sigma squared ML 
    Fstat    : tuple
               statistic (float), p-value (float)
    logll    : float
               Log likelihood        
    aic      : float
               Akaike info criterion 
    sc       : float
               Schwarz criterion     
    std_err  : array
               1*(k+1) array of Std.Error    
    Tstat    : list of tuples
               each tuple contains the pair (statistic, p-value), where each is
               a float; same order as self.x
    mulColli : float
               Multicollinearity condition number
    JB       : dictionary
               'jb': Jarque-Bera statistic (float); 'pvalue': p-value (float); 'df':
               degrees of freedom (int)  
    BP       : dictionary
               'bp': Breusch-Pagan statistic (float); 'pvalue': p-value (float); 'df':
               degrees of freedom (int)  
    KB       : dictionary
               'kb': Koenker-Bassett statistic (float); 'pvalue': p-value (float); 'df':
               degrees of freedom (int)  
    white    : dictionary
               'wh': White statistic (float); 'pvalue': p-value (float); 'df':
               degrees of freedom (int)  
    summary  : string
               including all the information in OLS class in nice format          
     
    
    Examples
    --------
    >>> import numpy as np
    >>> import pysal
    >>> db=pysal.open("examples/columbus.dbf","r")
    >>> y = np.array(db.by_col("CRIME"))
    >>> y = np.reshape(y, (49,1))
    >>> X = []
    >>> X.append(db.by_col("INC"))
    >>> X.append(db.by_col("HOVAL"))
    >>> X = np.array(X).T
    >>> ols = OLS(X, y, name_x=['INC','HOVAL'], name_y='CRIME', name_ds='columbus')
    >>> ols.betas
    array([[ 68.6189611 ],
           [ -1.59731083],
           [ -0.27393148]])
    
    """
    def __init__(self, x, y, constant=True, name_x=None, name_y=None,\
                        name_ds=None, vm=False, pred=False):
        OLS_dev.__init__(self, x, y, constant) 
        self.title = "ORDINARY LEAST SQUARES"        
        if not name_x:
            name_x = ['var_'+str(i+1) for i in range(len(x[0]))]
        if constant:
            name_x.insert(0, 'CONSTANT')
        if not name_y:
            name_y = 'dep_var'
        if not name_ds:
            name_ds = 'unknown'
        self.name_x = name_x
        self.name_ds = name_ds
        self.name_y = name_y
        USER.Diagnostic_Builder.__init__(self, constant=constant, vm=vm, pred=pred)


def _test():
    import doctest
    doctest.testmod()

if __name__ == '__main__':
    _test()



