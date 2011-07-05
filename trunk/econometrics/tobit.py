import numpy as np
import numpy.linalg as la
import scipy.optimize as op
import scipy.stats as stats
import scipy.sparse as SP
from math import pi
from probit import newton

class tobit:
    """
    Tobit class to do all the computations

    """
    def __init__(self,y,x,constant=True,w=None,optim='newton',maxiter=100):
        self.y = y        
        if constant:
            x = np.hstack((np.ones(y.shape),x))
        self.constant = constant
        self.x = x
        self.n, self.k = x.shape
        self.optim = optim
        self.w = w
        self.maxiter = maxiter
        self.n1 = int(sum(self.y>0))
        par_est, self.warning = self.par_est()
        self.sigma = 1/float(par_est[0][0])
        self.sigma2 = self.sigma**2
        self.gamma = np.array(par_est[0][1:]).reshape(self.k,1)
        self.betas = self.gamma*self.sigma
        self.logl = -float(par_est[1])-self.n1*(np.log(2*pi*self.sigma2))/2   
        self._cache = {}
        
    @property
    def vm(self):
        if 'vm' not in self._cache:
            nHinv = -la.inv(self.hessian(np.vstack((1/self.sigma,self.gamma))))
            J0 = np.vstack((-self.sigma2, np.zeros((self.k,1),float))).T
            J = np.vstack((J0,np.hstack((-self.sigma2*self.gamma,np.eye(self.k)*self.sigma))))
            self._cache['vm'] = np.dot(J,np.dot(nHinv,J.T))
        return self._cache['vm']    
    @property
    def Zstat(self):
        if 'Zstat' not in self._cache:
            variance = self.vm.diagonal()[1:]
            zStat = self.betas.reshape(len(self.betas),)/ np.sqrt(variance)
            rs = {}
            for i in range(len(self.betas)):
                rs[i] = (zStat[i],stats.norm.sf(abs(zStat[i]))*2)
            self._cache['Zstat'] = rs.values()
        return self._cache['Zstat']    
    @property
    def predy(self):
        if 'predy' not in self._cache:
            self.xb = np.dot(self.x,self.betas)
            xbs = self.xb/self.sigma
            self.cdfxbs = stats.norm.cdf(xbs)
            self._cache['predy'] = self.sigma*stats.norm.pdf(xbs)+self.cdfxbs*self.xb
        return self._cache['predy']
    @property
    def u(self):
        if 'u' not in self._cache:
            self._cache['u'] = self.y - self.predy
        return self._cache['u']
    @property
    def MoransI(self):
        if 'MoransI' not in self._cache: 
            if self.w:
                w = self.w.sparse
                moran_num = np.dot(self.u.T, (w * self.u))
                sig2i = self.predy*(self.xb-self.predy)+self.sigma2*self.cdfxbs
                E = SP.lil_matrix(w.get_shape())
                E.setdiag(sig2i.flat)
                E = E.asformat('csr')
                WE = w*E
                moran_den = np.sqrt(np.sum((WE*WE + (w.T*E)*WE).diagonal()))                
                moran = float(1.0*moran_num / moran_den)
                moran = np.array([moran,stats.norm.sf(abs(moran)) * 2.])                
            self._cache['MoransI'] = moran
        return self._cache['MoransI']
    
    def par_est(self):
        self.y1 = self.y[self.y>0].reshape(self.n1,1)       
        for i in range(self.k-1): #Must be recoded in a more efficient way.
            if i == 0:
                yk = np.hstack((self.y,self.y))
            else:
                yk = np.hstack((yk,self.y))
        self.x0 = self.x[yk==0].reshape((self.n-self.n1),self.k)
        self.x1 = self.x[yk>0].reshape(self.n1,self.k)
        ols1 = np.dot(la.inv(np.dot(self.x1.T,self.x1)),np.dot(self.x1.T,self.y1))
        x2t = np.hstack((np.ones(self.y1.shape),self.x1*np.dot(self.x1,ols1))).T
        xy1 = np.hstack((np.ones(self.y1.shape),self.x1*self.y1))
        start0 = np.dot(la.inv(np.dot(x2t,xy1)),np.dot(x2t,self.y1**2))
        start = np.vstack((1./np.sqrt(start0[0]),start0[1:]/np.sqrt(start0[0])))
        warn = 0
        flogl = lambda par: -self.ll(par)
        if self.optim == 'newton':            
            fgrad = lambda par: self.gradient(par)
            fhess = lambda par: self.hessian(par)
            par_hat = newton(flogl,start,fgrad,fhess,self.maxiter)
            warn = par_hat[2]
        else:
            fgrad = lambda par: -self.gradient(par)
            if self.optim == 'bfgs':
                par_hat = op.fmin_bfgs(flogl,start,fgrad,full_output=1,disp=1)
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
        gamma = par[1:].reshape(self.k,1)
        theta = par[0]
        ll0 = np.sum(np.log(stats.norm.cdf(-np.dot(self.x0,gamma))))
        ll1 = np.sum((theta*self.y1-np.dot(self.x1,gamma))**2)
        return ll0 - ll1/2.

    def gradient(self,par):        
        gamma = par[1:].reshape(self.k,1)
        theta = par[0] 
        x0g = np.dot(self.x0,gamma)
        x1g = np.dot(self.x1,gamma)
        mills = stats.norm.pdf(-x0g)/stats.norm.cdf(-x0g)
        gradG0 = np.dot(mills.T,self.x0)
        y1x1g = theta*self.y1 - x1g
        gradG = np.dot(y1x1g.T,self.x1) - gradG0
        gradT = self.n1/theta - np.dot(y1x1g.T,self.y1)
        return np.hstack((gradT,gradG))[0]

    def hessian(self,par):        
        gamma = par[1:].reshape(self.k,1)
        theta = par[0]
        theta2 = theta**2
        x0g = np.dot(self.x0,gamma)
        x1g = np.dot(self.x1,gamma)
        mills = stats.norm.pdf(-x0g)/stats.norm.cdf(-x0g)
        GG0 = np.dot((mills * (x0g - mills) * self.x0).T,self.x0)
        GG = GG0 - np.dot(self.x1.T,self.x1)
        GT = np.dot(self.x1.T,self.y1)
        TT = np.array(-self.n1/theta2 - np.sum(self.y1)).reshape(1,1)
        return np.vstack((np.hstack((TT,GT.T)),np.hstack((GT,GG))))

def _test():
    import doctest
    doctest.testmod()

if __name__ == '__main__':
    #_test()
    import numpy as np
    import pysal
    from econometrics.utils import power_expansion
    n = 1600
    x = np.random.uniform(-5,1,(n,1))
    x = np.hstack((np.ones(x.shape),x))
    xb = np.dot(x,np.reshape(np.array([1,0.5]),(2,1)))
    w = pysal.lat2W(int(np.sqrt(n)),int(np.sqrt(n)))
    w.transform='r'
    ys = xb + power_expansion(w, np.random.normal(0,1,(n,1)), 0.5) #Build y_{star}
    y = np.zeros((n,1),float) #Binary y
    for yi in range(len(y)):
        if ys[yi]>0:
            y[yi] = ys[yi] 
    tobit1=tobit(y,x,w=w,constant=False)#,optim='ncg'
    if tobit1.warning:
        print "Maximum number of iterations exceeded or gradient and/or function calls not changing."
    print "Variable  Coef.  S.E.  Z-Stat. p-Value"
    print "Constant %5.4f %5.4f %5.4f %5.4f" % (tobit1.betas[0],np.sqrt(tobit1.vm.diagonal())[1],tobit1.Zstat[0][0],tobit1.Zstat[0][1])
    for i in range(x.shape[1]-1):
        print "%-9s %5.4f %5.4f %5.4f %5.4f" % ('Var'+str([i]),tobit1.betas[i+1],np.sqrt(tobit1.vm.diagonal())[i+2],tobit1.Zstat[i+1][0],tobit1.Zstat[i+1][1])
    print "Sigma:", round(np.sqrt(float(tobit1.sigma2)),4)
    print "Log-Likelihood:", round(tobit1.logl,4)
    print "Moran's I:", round(tobit1.MoransI[0],3), "; pvalue:", round(tobit1.MoransI[1],4)
