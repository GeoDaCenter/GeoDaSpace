import numpy as np
import numpy.linalg as la
import scipy.optimize as op
from scipy.stats import norm, chisqprob
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
                  nx1 array of dependent binary variable
    constant    : boolean
                  If true it appends a vector of ones to the independent variables
                  to estimate intercept (set to True by default)
    w           : W
                  PySAL weights instance aligned with y
    optim       : string
                  Optimization method.
                  Default: 'newton' (Newton-Raphson).
                  Alternatives: 'ncg' (Newton-CG), 'bfgs' (BFGS algorithm)
    scalem      : string
                  Method to calculate the scale of the marginal effects.
                  Default: 'phimean' (Mean of individual marginal effects)
                  Alternative: 'xmean' (Marginal effects at variables mean)
    maxiter     : integer
                  Maximum number of iterations until optimizer stops                  
              
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
    logl        : float
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
    Pinkse_error: float
                  Lagrange Multiplier test against spatial error correlation.
                  Implemented as presented in Pinkse (2009)
    KP_error    : float
                  Moran's I type test against spatial error correlation.
                  Implemented as presented in Kelejian and Prucha (2001)
    PS_error    : float
                  Lagrange Multiplier test against spatial error correlation.
                  Implemented as presented in Pinkse and Slade (1998)
    warning     : boolean
                  if True Maximum number of iterations exceeded or gradient 
                  and/or function calls not changing.

    References
    ----------
    .. [1] Pinkse (2009)
    .. [2] Kelejian and Prucha (2001)
    .. [3] Pinkse and Slade (1998)

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
    >>> w = pysal.lat2W(8,4) #Optional spatial weights to run spatial tests 
    >>> w.transform='r'    
    >>> probit1=probit(y,X,scalem='xmean',w=w)
    >>> np.around(probit1.betas, decimals=3)
    array([[-7.452],
           [ 1.626],
           [ 0.052],
           [ 1.426]])
           
    >>> np.around(probit1.vm, decimals=2)
    array([[ 6.46, -1.17, -0.1 , -0.59],
           [-1.17,  0.48, -0.02,  0.11],
           [-0.1 , -0.02,  0.01,  0.  ],
           [-0.59,  0.11,  0.  ,  0.35]])

    >>> tests = np.array([['Pinkse_error','KP_error','PS_error','Pinkse_lag']])
    >>> stats = np.array([[probit1.Pinkse_error[0],probit1.KP_error[0],probit1.PS_error[0],probit1.Pinkse_lag[0]]])
    >>> pvalue = np.array([[probit1.Pinkse_error[1],probit1.KP_error[1],probit1.PS_error[1],probit1.Pinkse_lag[1]]])
    >>> print np.hstack((tests.T,np.around(np.hstack((stats.T,pvalue.T)),3)))
    [['Pinkse_error' '1.278' '0.258']
     ['KP_error' '-1.194' '0.232']
     ['PS_error' '0.725' '0.395']
     ['Pinkse_lag' '0.062' '0.803']]
    """
    def __init__(self,y,x,constant=True,w=None,optim='newton',scalem='phimean',maxiter=100):
        self.y = y        
        if constant:
            x = np.hstack((np.ones(y.shape),x))
        self.constant = constant
        self.x = x
        self.n, self.k = x.shape
        self.optim = optim
        self.scalem = scalem
        self.w = w
        self.maxiter = maxiter
        par_est, self.warning = self.par_est()
        self.betas = np.reshape(par_est[0],(self.k,1))
        self.logl = -float(par_est[1])
        self._cache = {}

    @property
    def vm(self):
        if 'vm' not in self._cache:
            H = self.hessian(self.betas)
            self._cache['vm'] = -la.inv(H)
        return self._cache['vm']
    @property
    def Zstat(self):
        if 'Zstat' not in self._cache:
            variance = self.vm.diagonal()
            zStat = self.betas.reshape(len(self.betas),)/ np.sqrt(variance)
            rs = {}
            for i in range(len(self.betas)):
                rs[i] = (zStat[i],norm.sf(abs(zStat[i]))*2)
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
            self._cache['predy'] = norm.cdf(self.xb)
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
            self._cache['phiy'] = norm.pdf(self.xb)
        return self._cache['phiy']
    @property
    def scale(self):
        if 'scale' not in self._cache:
            if self.scalem == 'phimean':
                self._cache['scale'] = float(1.0 * np.sum(self.phiy)/self.n)
            if self.scalem == 'xmean':
                self._cache['scale'] = float(norm.pdf(np.dot(self.xmean.T,self.betas)))
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
            P = 1.0 * np.sum(self.y) / self.n
            LR = float(-2 * (self.n*(P * np.log(P) + (1 - P) * np.log(1 - P)) - self.logl))  #Likeliood ratio test on all betas = zero.
            self._cache['LR'] = (LR,chisqprob(LR,self.k))
        return self._cache['LR']
    @property
    def u_naive(self):
        if 'u_naive' not in self._cache:
            u_naive = self.y - self.predy
            self._cache['u_naive'] = u_naive
        return self._cache['u_naive']
    @property
    def u_gen(self):
        if 'u_gen' not in self._cache:
            Phi_prod = self.predy * (1 - self.predy)
            u_gen = self.phiy * (self.u_naive / Phi_prod)
            self._cache['u_gen'] = u_gen
        return self._cache['u_gen']
    @property
    def Pinkse_error(self): #LM error and Moran's I tests are calculated together.
        if 'Pinkse_error' not in self._cache: 
            if self.w:
                w = self.w.sparse
                Phi = self.predy
                phi = self.phiy                
                #Pinkse_error:
                Phi_prod = Phi * (1 - Phi)
                u_naive = self.u_naive
                u_gen = self.u_gen
                sig2 = np.sum((phi * phi) / Phi_prod) / self.n
                LM_err_num = np.dot(u_gen.T,(w * u_gen))**2
                trWW = np.sum((w*w).diagonal())
                trWWWWp = trWW + np.sum((w*w.T).diagonal())
                LM_err = float(1.0 * LM_err_num / (sig2**2 * trWWWWp))
                LM_err = np.array([LM_err,chisqprob(LM_err,1)])
                #KP_error:
                moran = moran_KP(self.w,u_naive,Phi_prod)
                #Pinkse-Slade_error:
                u_std = u_naive / np.sqrt(Phi_prod)
                ps_num = np.dot(u_std.T, (w * u_std))**2
                trWpW = np.sum((w.T*w).diagonal())
                ps = float(ps_num / (trWW + trWpW))
                ps = np.array([ps,chisqprob(ps,1)]) #chi-square instead of bootstrap.
                #Pinkse_lag:
                Fn2 = np.dot((self.xb + u_gen).T,(w.T * u_gen))**2
                Jbb = la.inv(np.dot(self.x.T,self.x)*sig2)
                xmean = self.xmean
                muxb = np.dot(xmean.T,self.betas)
                W1 = w*np.ones((self.n,1))
                Jnb0 = muxb*sum(W1)*sig2
                LM_lag_den0 = np.dot(Jnb0*xmean.T,np.dot(Jbb,Jnb0*xmean))
                Jnn0 = trWpW*np.dot(self.betas.T,np.dot(np.cov(self.x,rowvar=0,bias=1),self.betas))
                Jnn = ((sig2*trWWWWp) + Jnn0 + ((muxb**2)*(la.norm(W1)**2)))*sig2
                LM_lag = Fn2 / (Jnn - LM_lag_den0)
                LM_lag = np.array([float(LM_lag),chisqprob(LM_lag,1)])
                self._cache['Pinkse_error'] = LM_err
                self._cache['KP_error'] = moran
                self._cache['PS_error'] = ps
                self._cache['Pinkse_lag'] = LM_lag
            else:
                raise Exception, "W matrix not provided to calculate spatial test."
        return self._cache['Pinkse_error']
    @property
    def KP_error(self): #All tests for spatial error correlation are calculated together.
        if 'KP_error' not in self._cache:
            self._cache['Pinkse_error'] = self.Pinkse_error
        return self._cache['KP_error']
    @property
    def Pinkse_lag(self): #All tests for spatial error correlation are calculated together.
        if 'Pinkse_lag' not in self._cache:
            self._cache['Pinkse_error'] = self.Pinkse_error
        return self._cache['Pinkse_lag']
    @property
    def PS_error(self): #All tests for spatial error correlation are calculated together.
        if 'PS_error' not in self._cache:
            self._cache['Pinkse_error'] = self.Pinkse_error
        return self._cache['PS_error']

    def par_est(self):
        start = np.dot(la.inv(np.dot(self.x.T,self.x)),np.dot(self.x.T,self.y))
        flogl = lambda par: -self.ll(par)
        if self.optim == 'newton':
            fgrad = lambda par: self.gradient(par)
            fhess = lambda par: self.hessian(par)            
            par_hat = newton(flogl,start,fgrad,fhess,self.maxiter)
            warn = par_hat[2]
        else:            
            fgrad = lambda par: -self.gradient(par)
            if self.optim == 'bfgs':
                par_hat = op.fmin_bfgs(flogl,start,fgrad,full_output=1,disp=0)
                warn = par_hat[6] 
            if self.optim == 'ncg':                
                fhess = lambda par: -self.hessian(par)
                par_hat = op.fmin_ncg(flogl,start,fgrad,fhess=fhess,full_output=1,disp=0)
                warn = par_hat[5]
        if warn > 0:
            warn = True
        else:
            warn = False
        return par_hat, warn

    def ll(self,par):       
        beta = np.reshape(np.array(par),(self.k,1))
        q = 2 * self.y - 1
        qxb = q * np.dot(self.x,beta)
        ll = sum(np.log(norm.cdf(qxb)))
        return ll

    def gradient(self,par):      
        beta = np.reshape(np.array(par),(self.k,1))
        q = 2 * self.y - 1
        qxb = q * np.dot(self.x,beta)
        lamb = q * norm.pdf(qxb)/norm.cdf(qxb)
        gradient = np.dot(lamb.T,self.x)[0]
        return gradient

    def hessian(self,par):           
        beta = np.reshape(np.array(par),(self.k,1))
        q = 2 * self.y - 1
        xb = np.dot(self.x,beta)
        qxb = q * xb
        lamb = q * norm.pdf(qxb)/norm.cdf(qxb)
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
            marg = self.betas[pos]*norm.pdf(float(x0*self.betas[pos] + pred))
            cumu = norm.cdf(float(x0*self.betas[pos] + pred))
            curves.append([x0,marg,cumu])
        return curves

#The following functions are to be moved to utils:
def newton(flogl,start,fgrad,fhess,maxiter):
    warn = 0
    iteration = 0
    par_hat0 = start
    m = 1
    while (iteration < maxiter and m>=1e-04):
        H = -la.inv(fhess(par_hat0))
        g = fgrad(par_hat0).reshape(start.shape)
        Hg = np.dot(H,g)
        par_hat0 = par_hat0 + Hg
        iteration += 1
        m = np.dot(g.T,Hg)
    if iteration == maxiter:
        warn = 1
    logl = flogl(par_hat0)
    return (par_hat0, logl, warn)

def moran_KP(w,u,sig2i):
    w = w.sparse
    moran_num = np.dot(u.T, (w * u))
    E = SP.lil_matrix(w.get_shape())
    E.setdiag(sig2i.flat)
    E = E.asformat('csr')
    WE = w*E
    moran_den = np.sqrt(np.sum((WE*WE + (w.T*E)*WE).diagonal()))      
    moran = float(1.0*moran_num / moran_den)
    moran = np.array([moran,norm.sf(abs(moran)) * 2.])
    return moran

def _test():
    import doctest
    doctest.testmod()

if __name__ == '__main__':
    _test()
    #Extended example:
    import numpy as np
    import pysal
    db = pysal.open("examples/greene21_1.csv",'r')
    var = {}
    var['y']="GRADE"
    var['x']=("GPA","TUCE","PSI")
    y = np.array(db.by_col(var['y']))
    y = np.reshape(y, (y.shape[0],1))
    X = []
    for i in var['x']:
        X.append(db.by_col(i))

    X = np.array(X).T
    w = pysal.lat2W(8,4) #Optional fictional weights matrix to run spatial tests
    w.transform='r'
    probit1=probit(y,X,scalem='xmean',w=w)
    if probit1.warning:
        print "Maximum number of iterations exceeded or gradient and/or function calls not changing."
    print "Dependent variable: GRADE"
    print "Variable  Coef.  S.E.  Z-Stat. p-Value Slope Slope_SD"
    print "Constant %5.4f %5.4f %5.4f %5.4f" % (probit1.betas[0],np.sqrt(probit1.vm.diagonal())[0],probit1.Zstat[0][0],probit1.Zstat[0][1])
    for i in range(len(var['x'])):
        print "%-9s %5.4f %5.4f %5.4f %5.4f %5.4f %5.4f" % (var['x'][i],probit1.betas[i+1],np.sqrt(probit1.vm.diagonal())[i+1],probit1.Zstat[i+1][0],probit1.Zstat[i+1][1],probit1.slopes[i],np.sqrt(probit1.slopes_vm.diagonal())[i])
    print "Log-Likelihood:", round(probit1.logl,4)
    print "LR test:", round(probit1.LR[0],3), "; pvalue:", round(probit1.LR[1],4)
    print "% correctly predicted:", round(probit1.predpc,2),"%"
    print "Spatial tests use fictional weights matrix:"
    print "Pinkse Spatial Error:", round(probit1.Pinkse_error[0],3), "; pvalue:", round(probit1.Pinkse_error[1],4)
    print "Pinkse Spatial Lag:", round(probit1.Pinkse_lag[0],3), "; pvalue:", round(probit1.Pinkse_lag[1],4)
    print "KP Spatial Error:", round(probit1.KP_error[0],3), "; pvalue:", round(probit1.KP_error[1],4)
    print "PS Spatial Error:", round(probit1.PS_error[0],3), "; pvalue:", round(probit1.PS_error[1],4)    