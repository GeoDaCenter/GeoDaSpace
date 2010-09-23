import sys
sys.path.append('C:/Users/Pedro/Documents/Academico/GeodaCenter/SVN/')
sys.path.append('C:/Users/Pedro/Documents/Academico/GeodaCenter/SVN/spreg/')
import pysal as pysal
import numpy as np
import numpy.linalg as la
import scipy.optimize as op
import scipy.stats as stats
import diagnostics as diagnostics
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
    slope       : string
                  Method to calculate the marginal effects.
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
    *u           : array
                  nx1 array of residuals (Yet to be coded!)
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
    scale       : float
                  Scale of the marginal effects
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
    scale       : float
                  Scale of the marginal effects

    References
    ----------
    .. [1] Pinkse (2009)
    .. [2] Kelejian and Prucha (2001)

    Examples
    --------

   
    """
    def __init__(self,spec,constant=True,w=None,optim='newton',slope='phimean'):
        db = spec['data']
        xname = spec['x_label']
        yname = spec['y_label']
        x = []
        for i in xname:
            x.append(db.by_col(i))
        x = np.array(x).T
        y = np.array(db.by_col(yname[0]))
        self.y = np.reshape(y, (y.shape[0],1))
        if constant:
            x = np.hstack((np.ones(self.y.shape),x))
        self.constant = constant
        self.x = x
        self.n, self.k = x.shape
        par_est = self.par_est(optim)
        self.betas = np.reshape(par_est[0],(self.k,1))
        self.logll = float(par_est[1])
        self.vm = self.cov_matrix(self.betas)
        #self.Tstat = diagnostics.t_stat(self) #For Probit the normal stand. dist. should be used instead of the t.
        self.Tstat = self.z_stat()
        self.xmean = np.reshape(sum(self.x)/self.n,(self.k,1))
        self.predy = stats.norm.cdf(np.dot(self.x,self.betas))
        self.predpc = float(self.get_predpc())
        phi = np.reshape(np.array(stats.norm.pdf(self.predy)),(self.n,1))
        if slope == 'phimean':
            self.scale = 1.0 * sum(phi)/self.n
        if slope == 'xmean':
            self.scale = stats.norm.pdf(np.dot(self.xmean.T,self.betas))
        self.slopes = self.betas[1:] * self.scale #Disregard the presence of dummies.
        self.slopes_vm = self.get_slopes_vm(self.xmean)
        P = 1.0 * sum(self.y)[0] / self.n
        LR = float(-2 * (self.n*(P * np.log(P) + (1 - P) * np.log(1 - P)) - self.logll))  #Likeliood ratio test on all betas = zero.
        self.LR = (LR,stats.chisqprob(LR,self.k))
        if w: 
            self.LM_error, self.moran = self.get_spatial_tests(w,phi)

    def par_est(self,optim):
        start = []
        for i in range(self.x.shape[1]):
            start.append(0.01)
        flogl = lambda par: -self.ll(par)
        fgrad = lambda par: -self.gradient(par)
        fhess = lambda par: -self.hessian(par)
        if optim == 'newton':
            iteration = 0
            start = np.array(start)
            history = [np.inf, start]
            while (iteration < 50 and np.all(np.abs(history[-1] - history[-2])>1e-08)):
                H = self.hessian(history[-1])
                par_hat0 = history[-1] - np.dot(np.linalg.inv(H),self.gradient(history[-1]))
                history.append(par_hat0)
                iteration += 1
            logl = self.ll(par_hat0,final=1)
            par_hat = [] #Coded like this to comply with most of the scipy optimizers.
            par_hat.append(par_hat0)
            par_hat.append(logl)            
        return par_hat

    def cov_matrix(self,par):
        H = self.hessian(par,final=1)
        Hinv = -la.inv(H)
        return Hinv

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

    def get_spatial_tests(self,w,phi):
        Phi = np.reshape(np.array(stats.norm.cdf(self.predy)),(self.n,1))
        Phi_prod = Phi * (1 - Phi)
        u_naive = self.y - Phi
        u = phi * (u_naive / Phi_prod)
        sig2 = sum((phi * phi) / Phi_prod) / self.n
        LM_err_num = np.dot((u.T * w.sparse), u)**2
        LM_err_den = sig2**2 * np.sum(((w.sparse*w.sparse)+(w.sparse.T*w.sparse)).diagonal())
        LM_err = float(LM_err_num / LM_err_den)
        LM_err = np.array([LM_err,stats.chisqprob(LM_err,1)])
        E = SP.lil_matrix(w.sparse.get_shape()) #There's a similar code in gmm_utils to create the Sigma matrix for the Psi.
        E.setdiag(Phi.flat)
        E = E.asformat('csr')
        WE = w.sparse*E
        moran_den =  np.sqrt(np.sum((WE*WE + (w.sparse.T*E)*WE).diagonal()))
        moran = float(np.dot((u_naive.T * w.sparse), u_naive))
        moran = np.array([moran,stats.norm.sf(abs(moran)) * 2.])
        return LM_err, moran

    def z_stat(self): #Should be in spreg.diagnostics
        variance = self.vm.diagonal()
        zStat = self.betas.reshape(len(self.betas),)/ np.sqrt(variance)
        rs = {}
        for i in range(len(self.betas)):
            rs[i] = (zStat[i],stats.norm.sf(abs(zStat[i]))*2)
        ts_result  = rs.values()
        return ts_result

    def get_slopes_vm(self,x):
        b = self.betas
        dfdb = np.eye(self.k) - np.dot(b.T,x)*np.dot(b,x.T)
        slopes_vm = (self.scale**2)*np.dot(np.dot(dfdb,self.vm),dfdb.T)
        return slopes_vm[1:,1:]

    def get_predpc(self):
        predpc = abs(self.y-self.predy)
        for i in range(len(predpc)):
            if predpc[i]>0.5:
                predpc[i]=0
            else:
                predpc[i]=1
        return 100* sum(predpc) / self.n

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
    
if __name__ == '__main__':
    # Must be transformed in an example.
                
    spec = {}
    spec['data'] = pysal.open("examples/greene21_1.csv",'r')
    spec['x_label'] = ['GPA','TUCE','PSI']
    spec['y_label'] = ['GRADE']
    #w = pysal.open('fake.gal', 'r').read()
    #w.transform='r'
    #probit1=probit(spec,w=w)
    probit1=probit(spec,slope='xmean')
    print "Parameters:"
    print probit1.betas
    print "Std-Dev:"
    print np.sqrt(probit1.vm.diagonal())
    print "Z stat:" 
    print probit1.Tstat
    print "Slopes:" 
    print probit1.slopes
    print "Slopes_SD:" 
    print np.sqrt(probit1.slopes_vm.diagonal())
    print "Log-Likelihood:" 
    print probit1.logll
    print "LR test:" 
    print probit1.LR
    print "% correctly predicted:" 
    print probit1.predpc
    #curve = np.reshape(np.array(probit1.slope_graph(1,10,sample='actual')),(10,3))
    #print curve
    #print "LM Error - Pinkse (1999):"
    #print probit1.LM_error
    #print "Moran's I - KP (2001):"
    #print probit1.moran
