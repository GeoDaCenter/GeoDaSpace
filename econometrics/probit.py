import pysal as pysal
import numpy as np
import numpy.linalg as la
import scipy.optimize as op
import scipy.stats as stats
import scipy.sparse as SP

class probit: #DEV class required.
    # Must add doc on slope_graph()
    """
    Probit class to do all the computations

    Parameters
    ----------

    x           : array
                  nxk array of independent variables (assumed to be aligned with y)
    y           : array
                  nx1 array of dependent variable
    constant    : boolean
                  If true it appends a vector of ones to the independent variables
                  to estimate intercept (set to True by default)
    w           : W
                  PySAL weights instance aligned with y
    optim       : string
                  Optimization method.
                  Default: 'newton' (Newton-Raphson).
                  Alternative: 'bhhh' (yet to be coded)
    scalem      : string
                  Method to calculate the scale of the marginal effects.
                  Default: 'phimean' (Mean of individual marginal effects)
                  Alternative: 'xmean' (Marginal effects at variables mean)
              
    Attributes
    ----------

    x           : array
                  nxk array of independent variables (assumed to be aligned with y)
    y           : array
                  nx1 array of dependent variable
    betas       : array
                  kx1 array with estimated coefficients
    predy       : array
                  nx1 array of predicted values
    n           : int
                  Number of observations
    k           : int
                  Number of variables
    constant    : boolean
                  Denotes if a constant is included in the regression
    vm          : array
                  Variance-covariance matrix (kxk)
    xmean       : array
                  Mean of the independent variables (kx1)
    predpc      : float
                  Percent of y correctly predicted
    logll       : float
                  Log-Likelihhod of the estimation
    *Tstat       : dictionary
                  key: name of variables, constant & independent variables (Yet to be coded!)
                  value: tuple of t statistic and p-value
    scalem      : string
                  Method to calculate the scale of the marginal effects.
    scale       : float
                  Scale of the marginal effects.
    slopes      : array
                  Marginal effects of the independent variables (k-1x1)
    slopes_vm   : array
                  Variance-covariance matrix of the slopes (k-1xk-1)
    LR          : tupe
                  Likelihood Ratio test of all coefficients = 0
                  (test statistics, p-value)
    LM_error    : float
                  Lagrange Multiplier test against spatial error correlation.
                  Implemented as presented in Pinkse (2009)
    moran       : float
                  Moran's I type test against spatial error correlation.
                  Implemented as presented in Kelejian and Prucha (2001)

    References
    ----------
    .. [1] Pinkse (2009)
    .. [2] Kelejian and Prucha (2001)

    Examples
    --------
    >>> import numpy as np
    >>> import pysal
    >>> db = pysal.open("examples/greene21_1.csv",'r')
    >>> y = np.array(db.by_col("GRADE"))
    >>> y = np.reshape(y, (y.shape[0],1))
    >>> X = []
    >>> X.append(db.by_col("GPA"))
    >>> X.append(db.by_col("TUCE"))
    >>> X.append(db.by_col("PSI"))
    >>> X = np.array(X).T
    >>> probit1=probit(X,y,scalem='xmean')
    >>> np.around(probit1.betas, decimals=3)
    array([[-7.452],
           [ 1.626],
           [ 0.052],
           [ 1.426]])
   
    """
    def __init__(self,x,y,constant=True,w=None,optim='newton',scalem='phimean'):
        self.y = y        
        if constant:
            x = np.hstack((np.ones(y.shape),x))
        self.constant = constant
        self.x = x
        self.n, self.k = x.shape
        self.optim = optim
        self.scalem = scalem
        self.w = w
        par_est = self.par_est()
        self.betas = np.reshape(par_est[0],(self.k,1))
        self.logll = float(par_est[1])
        self._cache = {}

    @property
    def vm(self):
        if 'vm' not in self._cache:
            H = self.hessian(self.betas,final=1)
            self._cache['vm'] = -la.inv(H)
        return self._cache['vm']
    @property
    def Zstat(self):
        if 'Zstat' not in self._cache:
            variance = self.vm.diagonal()
            zStat = self.betas.reshape(len(self.betas),)/ np.sqrt(variance)
            rs = {}
            for i in range(len(self.betas)):
                rs[i] = (zStat[i],stats.norm.sf(abs(zStat[i]))*2)
            self._cache['Zstat'] = rs.values()
        return self._cache['Zstat']
    @property
    def xmean(self):
        if 'xmean' not in self._cache:
            self._cache['xmean'] = np.reshape(sum(self.x)/self.n,(self.k,1))
        return self._cache['xmean']
    @property
    def xb(self):
        if 'xb' not in self._cache:
            self._cache['xb'] = np.dot(self.x,self.betas)
        return self._cache['xb']
    @property
    def predy(self):
        if 'predy' not in self._cache:
            self._cache['predy'] = stats.norm.cdf(self.xb)
        return self._cache['predy']
    @property
    def predpc(self):
        if 'predpc' not in self._cache:
            predpc = abs(self.y-self.predy)
            for i in range(len(predpc)):
                if predpc[i]>0.5:
                    predpc[i]=0
                else:
                    predpc[i]=1
            self._cache['predpc'] = float(100* np.sum(predpc) / self.n)
        return self._cache['predpc']
    @property
    def phiy(self):
        if 'phiy' not in self._cache:
            self._cache['phiy'] = stats.norm.pdf(self.xb)
        return self._cache['phiy']
    @property
    def scale(self):
        if 'scale' not in self._cache:
            if self.scalem == 'phimean':
                self._cache['scale'] = float(1.0 * np.sum(self.phiy)/self.n)
            if self.scalem == 'xmean':
                self._cache['scale'] = float(stats.norm.pdf(np.dot(self.xmean.T,self.betas)))
        return self._cache['scale']
    @property
    def slopes(self):
        if 'slopes' not in self._cache:
            self._cache['slopes'] = self.betas[1:] * self.scale #Disregard the presence of dummies.
        return self._cache['slopes']
    @property
    def slopes_vm(self):
        if 'slopes_vm' not in self._cache:
            x = self.xmean
            b = self.betas
            dfdb = np.eye(self.k) - np.dot(b.T,x)*np.dot(b,x.T)
            slopes_vm = (self.scale**2)*np.dot(np.dot(dfdb,self.vm),dfdb.T)
            self._cache['slopes_vm'] = slopes_vm[1:,1:]
        return self._cache['slopes_vm']
    @property
    def LR(self):
        if 'LR' not in self._cache:    
            P = 1.0 * np.sum(self.y)[0] / self.n
            LR = float(-2 * (self.n*(P * np.log(P) + (1 - P) * np.log(1 - P)) - self.logll))  #Likeliood ratio test on all betas = zero.
            self._cache['LR'] = (LR,stats.chisqprob(LR,self.k))
        return self._cache['LR']
    @property
    def LM_error(self): #LM error and Moran's I tests are calculated together.
        if 'LM_error' not in self._cache: 
            if self.w:
                w = self.w
                phi = self.phiy
                Phi = self.predy
                #LM_error:
                Phi_prod = Phi * (1 - Phi)
                u_naive = self.y - Phi
                u_gen = phi * (u_naive / Phi_prod)
                sig2 = np.sum((phi * phi) / Phi_prod) / self.n
                LM_err_num = np.dot(u_gen.T,(w.sparse * u_gen))**2
                LM_err_den = sig2**2 * np.sum(((w.sparse*w.sparse)+(w.sparse.T*w.sparse)).diagonal())
                LM_err = float(1.0 * LM_err_num / LM_err_den)
                LM_err = np.array([LM_err,stats.chisqprob(LM_err,1)])
                #Moran's I:
                E = SP.lil_matrix(w.sparse.get_shape()) #There's a similar code in gmm_utils to create the Sigma matrix for the Psi.
                E.setdiag(Phi_prod.flat)
                E = E.asformat('csr')
                WE = w.sparse*E
                moran_den = np.sqrt(np.sum((WE*WE + (w.sparse.T*E)*WE).diagonal()))
                moran_num = np.dot(u_naive.T, (w.sparse * u_naive))
                moran = float(1.0*moran_num / moran_den)
                moran = np.array([moran,stats.norm.sf(abs(moran)) * 2.])
                self._cache['LM_error'] = LM_err
                self._cache['moran'] = moran
            else:
                print "W not specified."
        return self._cache['LM_error']
    @property
    def moran(self): #LM error and Moran's I tests are calculated together.
        if 'moran' not in self._cache:
            self._cache['LM_error'] = self.LM_error
            self._cache['moran'] = self.moran
        return self._cache['moran']

    def par_est(self):
        start = []
        for i in range(self.x.shape[1]):
            start.append(0.01)
        flogl = lambda par: -self.ll(par)
        fgrad = lambda par: -self.gradient(par)
        fhess = lambda par: -self.hessian(par)
        if self.optim == 'newton':
            iteration = 0
            start = np.array(start)
            history = [np.inf, start]
            while (iteration < 50 and np.all(np.abs(history[-1] - history[-2])>1e-08)):
                H = self.hessian(history[-1])
                par_hat0 = history[-1] - np.dot(np.linalg.inv(H),self.gradient(history[-1]))
                history.append(par_hat0)
                iteration = iteration + 1
            logl = self.ll(par_hat0,final=1)
            par_hat = [] #Coded like this to comply with most of the scipy optimizers.
            par_hat.append(par_hat0)
            par_hat.append(logl)            
        return par_hat

    def ll(self,par,final=None):
        if final:
            beta = par
        else:
            beta = []
            for i in range(self.k):
                beta.append(float(par[i]))         
        beta = np.reshape(np.array(beta),(self.k,1))
        q = 2 * self.y - 1
        qxb = q * np.dot(self.x,beta)
        ll = sum(np.log(stats.norm.cdf(qxb)))
        return ll

    def gradient(self,par):        
        beta = []
        for i in range(self.k):
            beta.append(float(par[i]))         
        beta = np.reshape(np.array(beta),(self.k,1))
        q = 2 * self.y - 1
        qxb = q * np.dot(self.x,beta)
        lamb = q * stats.norm.pdf(qxb)/stats.norm.cdf(qxb)
        gradient = np.dot(lamb.T,self.x)[0]
        return gradient

    def hessian(self,par,final=None):        
        if final:
            beta = par
        else:
            beta = []
            for i in range(self.k):
                beta.append(float(par[i]))             
        beta = np.reshape(np.array(beta),(self.k,1))
        q = 2 * self.y - 1
        xb = np.dot(self.x,beta)
        qxb = q * xb
        lamb = q * stats.norm.pdf(qxb)/stats.norm.cdf(qxb)
        hessian = np.dot((self.x.T),(-lamb * (lamb + xb) * self.x ))
        return hessian

    def slope_graph(self,pos,k,sample='actual',std=2):
        if sample=='actual':
            assert k<=self.n, "k must not be more than the number of observations."            
            nk = 1.0*(self.n-1) / (k-1)
            x = sorted(self.x[:,pos])
        if sample=='input':
            nk = 1.0 / (k-1)
            xave = np.mean(self.x[:,pos])
            xstd = np.std(self.x[:,pos])
        pred = self.xmean*self.betas
        pred = sum(np.vstack((pred[0:pos],pred[(pos+1):])))
        curves = []
        for i in range(k):
            if sample=='actual':
                x0 = x[int(round(i*nk))]
            if sample=='input':
                x0 = xave + (i*nk - 0.5)*std*2*xstd
            marg = self.betas[pos]*stats.norm.pdf(float(x0*self.betas[pos] + pred))
            cumu = stats.norm.cdf(float(x0*self.betas[pos] + pred))
            curves.append([x0,marg,cumu])
        return curves

def _test():
    import doctest
    doctest.testmod()

if __name__ == '__main__':
    _test()
