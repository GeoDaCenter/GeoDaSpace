import numpy as np
import numpy.linalg as la


class OLS_dev:
    """
    OLS class to do all the computations

    NOTE: no consistency checks
    
    Maximal Complexity: O(n^3)

    Parameters
    ----------

    x       : array
              nxk array of independent variables (assumed to be aligned with y)
    y       : array
              nx1 array of dependent variable
    constant: boolean
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

              .. math::
                
                    \sigma^2 = \dfrac{\tilde{u}' \tilde{u}}{N}
    m       : array
              Matrix M

              .. math::

                    M = I - X(X'X)^{-1}X'


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
        xt = np.transpose(x)
        xtx = np.dot(xt,x)
        xtxi = la.inv(xtx)
        xty = np.dot(xt,y)
        self.betas = np.dot(xtxi,xty)
        self.xtxi = xtxi
        self.xtx = xtx
        self.xt = xt
        predy = np.dot(x,self.betas)
        u = y-predy
        self.u = u
        self.predy = predy
        self.y = y
        self.x = x
        self.n, self.k = x.shape
        self._cache = {}

    @property
    def utu(self):
        if 'utu' not in self._cache:
            self._cache['utu'] = np.sum(self.u**2)
        return self._cache['utu']
    @property
    def sig2(self):
        if 'sig2' not in self._cache:
            self._cache['sig2'] = self.utu / self.n
        return self._cache['sig2']
    @property
    def sig2n_k(self):
        if 'sig2' not in self._cache:
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
            estSig2= self.utu / (self.n-self.k)
            self._cache['vm'] = np.dot(estSig2, self.xtxi)
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
    

import diagnostics

class OLS:
    """
    OLS class for end-user (gives back only results and diagnostics)
    
    Maximal Complexity: O(n^3)

    Parameters
    ----------

    x       : array
              nxk array of independent variables (assumed to be aligned with y)
    y       : array
              nx1 array of dependent variable
    names   : tuple
              used in summary output, the sequence is (dataset name, dependent name, independent names)
    
    Attributes
    ----------

    x       : array
              nxk array of independent variables (assumed to be aligned with y)
    y       : array
              nx1 array of dependent variable
    betas   : array
              kx1 array with estimated coefficients
    u       : array
              nx1 array of residuals
    predy   : array
              nx1 array of predicted values
    n       : int
              Number of observations
    k       : int
              Number of variables
    name_ds : string
              dataset's name
    name_y  : string
              dependent variable's name
    name_x  : tuple
              independent variables' names
    mean_y  : float
              mean value of dependent variable
    std_y   : float
              standard deviation of dependent variable
    vm      : variance covariance matrix
              k*k array
    r2      : float
              R square
    ar2     : float
              adjusted R square
    utu     : float
              Sum of the squared residuals
    sig2    : float
              sigma squared
    sig2ML  : float
              sigma squared ML 
    Fstat   : dictionary
              key: 'value','prob'; value: float
    logll   : float
              Log likelihood        
    aic     : float
              Akaike info criterion 
    sc      : float
              Schwarz criterion     
    std_err : array
              1*(k+1) array of Std.Error    
    Tstat   : dictionary
              key: name of variables, constant & independent variables
              value: tuple of t statistic and p-value
    mulColli: float
              Multicollinearity condition number
    diag    : dictionary
              key: test name including 'JB','BP','KB','WH',representing "Jarque-Bera","Breusch-Pagan",
              "Koenker-Bassett","White"
              value: tuple including 3 elements--degree of freedom, value, p-value
    summary : print all the information in OLS class in nice format          
     
    
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
    >>> ols=OLS(X,y,('columbus,'CRIME','INC','HOVAL'))
    >>> ols.betas
    array([[ 68.6189611 ],
           [ -1.59731083],
           [ -0.27393148]])
    
    """
    def __init__(self,x,y,names):
        ols = OLS_dev(x,y) 
        
        #part 1: ORDINARY LEAST SQUARES ESTIMATION 
        
        #general information
        self.x = ols.x
        self.y = ols.y
        self.betas = ols.betas
        self.u = ols.u
        self.predy = ols.predy
        self.n = ols.n
        self.k = ols.k  
        self.name_ds = names[0]
        self.name_y = names[1]
        self.name_x = names[2:]
        self.mean_y = ols.mean_y
        self.std_y = ols.std_y
        self.vm = ols.vm
        self.utu = ols.utu
        self.sig2 = ols.sig2n_k
        
        self.r2 = diagnostics.r2_ols(self)    
        self.ar2 = diagnostics.ar2_ols(self)   
        self.sigML = ols.sig2  
        self.Fstat = diagnostics.fStat_ols(self)  
        self.logll = diagnostics.LogLikelihood(ols) 
        self.aic = diagnostics.AkaikeCriterion(ols) 
        self.sc = diagnostics.SchwarzCriterion(ols) 
        
        #Coefficient, Std.Error, t-Statistic, Probability 
        self.std_err = diagnostics.stdError_Betas(self)
        self.Tstat = diagnostics.tStat_ols(self)
        
        #part 2: REGRESSION DIAGNOSTICS 
        self.mulColli = diagnostics.MultiCollinearity(ols)
        self.diag = {}
        self.diag['JB'] = diagnostics.JarqueBera(ols)
        
        #part 3: DIAGNOSTICS FOR HETEROSKEDASTICITY         
        self.diag['BP'] = diagnostics.BreuschPagan(ols)
        self.diag['KB'] = {'df':2,'kb':5.694088,'pvalue':0.0580156}
        self.diag['WH'] = {'df':5,'wh':19.94601,'pvalue':0.0012792}
        
        #part 4: COEFFICIENTS VARIANCE MATRIX
        self.vm = ols.vm
        
        #part 5: summary output
        self.summary = output_ols(self)



def _test():
    import doctest
    doctest.testmod()

if __name__ == '__main__':
    _test()



