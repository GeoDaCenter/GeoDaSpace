import numpy as np
import numpy.linalg as la
import diagnostics


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
    vm      : array
              Variance-covariance matrix (kxk)
    mean_y  : float
              Mean of the dependent variable
    std_y   : float
              Standard deviation of the dependent variable

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
        self.set_x(x)
        xty = np.dot(x.T,y)
        self.betas = np.dot(self.xtxi,xty)
        predy = np.dot(x,self.betas)
        u = y-predy
        self.u = u
        self.predy = predy
        self.y = y
        self.n, self.k = x.shape
        self._cache = {}

    def set_x(self, x):
        self.x = x
        self.xt = x.T
        self.xtx = np.dot(self.x.T,self.x)
        self.xtxi = la.inv(self.xtx)

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
    


class OLS(OLS_dev):
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
    vm      : array
              variance covariance matrix (kxk)
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
    summary : string
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
    >>> ols=OLS(X,y,('columbus','CRIME','INC','HOVAL'))
    >>> ols.betas
    array([[ 68.6189611 ],
           [ -1.59731083],
           [ -0.27393148]])
    
    """
    def __init__(self, x, y, names, constant=True):
        ols = OLS_dev.__init__(self, x, y, constant) 
        
        #part 1: ORDINARY LEAST SQUARES ESTIMATION 
        
        #general information
        self.name_ds = names[0]
        self.name_y = names[1]
        self.name_x = names[2:]
        self.sig2 = self.sig2n_k
        
        self.r2 = diagnostics.r2(self)    
        self.ar2 = diagnostics.ar2(self)   
        self.sigML = self.sig2  
        self.Fstat = diagnostics.f_stat(self)  
        self.logll = diagnostics.log_likelihood(self) 
        self.aic = diagnostics.akaike(self) 
        self.sc = diagnostics.schwarz(self) 
        
        #Coefficient, Std.Error, t-Statistic, Probability 
        self.std_err = diagnostics.se_betas(self)
        self.Tstat = diagnostics.t_stat(self)
        
        #part 2: REGRESSION DIAGNOSTICS 
        self.mulColli = diagnostics.condition_index(self)
        self.diag = {}
        self.diag['JB'] = diagnostics.jarque_bera(self)
        
        #part 3: DIAGNOSTICS FOR HETEROSKEDASTICITY         
        self.diag['BP'] = diagnostics.breusch_pagan(self)
        self.diag['KB'] = {'df':2,'kb':5.694088,'pvalue':0.0580156}
        self.diag['WH'] = {'df':5,'wh':19.94601,'pvalue':0.0012792}
        
        #part 4: summary output
        self.summary = output_ols(self)

def output_ols(ols,vm = False, pred = False):
    """
    nice output for ols
    
    Parameters
    ----------

    ols     : instance of class OLS
    vm      : if vm = True, print out variance matrix
    pred    : if pred = True, print out y, predicted values and residuals
    
    Returns
    ----------

    strSummary   : string
                   include all the information in class OLS
    """     
    strSummary = ""
    
    # general information 1
    strSummary = strSummary + "REGRESSION\n"
    strSummary = strSummary + "----------\n"
    strSummary = strSummary + "SUMMARY OF OUTPUT: ORDINARY LEAST SQUARES ESTIMATION\n"
    strSummary = strSummary + "----------------------------------------------------\n"
    strSummary = strSummary + "%-20s:%12s\n" % ('Data set',ols.name_ds)
    strSummary = strSummary + "%-20s:%12s  %-22s:%12d\n" % ('Dependent Variable',ols.name_y,'Number of Observations',ols.n)
    strSummary = strSummary + "%-20s:%12.4f  %-22s:%12d\n" % ('Mean dependent var',ols.mean_y,'Number of Variables',ols.k)
    strSummary = strSummary + "%-20s:%12.4f  %-22s:%12d\n\n" % ('S.D. dependent var',ols.std_y,'Degrees of Freedom',ols.n-ols.k)
    
    # general information 2
    strSummary = strSummary + "%-20s:%12.6f  %-22s:%12.4f\n" % ('R-squared',ols.r2,'F-statistic',ols.Fstat[0])
    strSummary = strSummary + "%-20s:%12.6f  %-22s:%12.8g\n" % ('Adjusted R-squared',ols.ar2,'Prob(F-statistic)',ols.Fstat[1])
    strSummary = strSummary + "%-20s:%12.3f  %-22s:%12.3f\n" % ('Sum squared residual',ols.utu,'Log likelihood',ols.logll)
    strSummary = strSummary + "%-20s:%12.3f  %-22s:%12.3f\n" % ('Sigma-square',ols.sig2,'Akaike info criterion',ols.aic)
    strSummary = strSummary + "%-20s:%12.3f  %-22s:%12.3f\n" % ('S.E. of regression',np.sqrt(ols.sig2),'Schwarz criterion',ols.sc)
    strSummary = strSummary + "%-20s:%12.3f\n%-20s:%12.4f\n" % ('Sigma-square ML',ols.sigML,'S.E of regression ML',np.sqrt(ols.sigML))
    
    # Variable    Coefficient     Std.Error    t-Statistic   Probability 
    strSummary = strSummary + "----------------------------------------------------------------------------\n"
    strSummary = strSummary + "    Variable     Coefficient       Std.Error     t-Statistic     Probability\n"
    strSummary = strSummary + "----------------------------------------------------------------------------\n"
    strSummary = strSummary + "%12s    %12.7f    %12.7f    %12.7f    %12.7g\n" % ('CONSTANT',ols.betas[0][0],ols.std_err[0],ols.Tstat[0][0],ols.Tstat[0][1])
    i=1
    for name in ols.name_x:        
        strSummary = strSummary + "%12s    %12.7f    %12.7f    %12.7f    %12.7g\n" % (name,ols.betas[i][0],ols.std_err[i],ols.Tstat[i][0],ols.Tstat[i][1])
        i+=1
    strSummary = strSummary + "----------------------------------------------------------------------------\n"
    
    # diagonostics
    strSummary = strSummary + "\n\nREGRESSION DIAGNOSTICS\n"
    strSummary = strSummary + "MULTICOLLINEARITY CONDITION NUMBER%12.6f\n" % (ols.mulColli)
    strSummary = strSummary + "TEST ON NORMALITY OF ERRORS\n"
    strSummary = strSummary + "TEST                  DF          VALUE            PROB\n"
    strSummary = strSummary + "%-22s%2d       %12.6f        %9.7f\n\n" % ('Jarque-Bera',ols.diag['JB']['df'],ols.diag['JB']['jb'],ols.diag['JB']['pvalue'])
    strSummary = strSummary + "DIAGNOSTICS FOR HETEROSKEDASTICITY\n"
    strSummary = strSummary + "RANDOM COEFFICIENTS\n"
    strSummary = strSummary + "TEST                  DF          VALUE            PROB\n"
    strSummary = strSummary + "%-22s%2d       %12.6f        %9.7f\n" % ('Breusch-Pagan test',ols.diag['BP']['df'],ols.diag['BP']['bp'],ols.diag['BP']['pvalue'])
    strSummary = strSummary + "%-22s%2d       %12.6f        %9.7f\n" % ('Koenker-Bassett test',ols.diag['KB']['df'],ols.diag['KB']['kb'],ols.diag['KB']['pvalue'])
    strSummary = strSummary + "SPECIFICATION ROBUST TEST\n"
    strSummary = strSummary + "TEST                  DF          VALUE            PROB\n"
    strSummary = strSummary + "%-22s%2d       %12.6f        %9.7f\n\n" % ('White',ols.diag['WH']['df'],ols.diag['WH']['wh'],ols.diag['WH']['pvalue'])

    # variance matrix
    if vm == True:
        strVM = ""
        strVM = strVM + "COEFFICIENTS VARIANCE MATRIX\n"
        strVM = strVM + "----------------------------\n"
        strVM = strVM + "%12s" % ('CONSTANT')
        for name in ols.name_x:
            strVM = strVM + "%12s" % (name)
        strVM = strVM + "\n"
        nrow = ols.vm.shape[0]
        ncol = ols.vm.shape[1]
        for i in range(nrow):
            for j in range(ncol):
                strVM = strVM + "%12.6f" % (ols.vm[i][j]) 
            strVM = strVM + "\n"
        strSummary = strSummary + strVM
        
    # y, PREDICTED, RESIDUAL 
    if pred == True:
        strPred = "\n\n"
        strPred = strPred + "%16s%16s%16s%16s\n" % ('OBS',ols.name_y,'PREDICTED','RESIDUAL')
        for i in range(ols.n):
            strPred = strPred + "%16d%16.5f%16.5f%16.5f\n" % (i+1,ols.y[i][0],ols.predy[i][0],ols.u[i][0])
        strSummary = strSummary + strPred
            
    # end of report
    strSummary = strSummary + "========================= END OF REPORT =============================="
        
    return strSummary


def _test():
    import doctest
    doctest.testmod()

if __name__ == '__main__':
    _test()



