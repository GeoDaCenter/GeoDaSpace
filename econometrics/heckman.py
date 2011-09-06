import numpy as np
import numpy.linalg as la
#import scipy.optimize as op
from scipy.stats import norm
#import scipy.sparse as SP
#from math import pi
import time
from probit import probit, newton, moran_KP
from pysal.spreg.ols import BaseOLS

class heckman:
    """
    Tobit class to do all the computations

    """
    def __init__(self,y1,x1,x2,constant1=True,constant2=True,twostep=False,w=None,optim='newton',maxiter=100):
        self.y1 = y1        
        if constant1:
            x1 = np.hstack((np.ones(y1.shape),x1))
        self.constant1 = constant1
        self.x1 = x1
        self.n, self.k1 = x1.shape
        if constant2:
            x2 = np.hstack((np.ones((x2.shape[0],1)),x2))
        self.constant2 = constant2
        self.x2 = x2
        self.k2 = x2.shape[1]        
        self.twostep = twostep
        self.optim = optim
        self.w = w
        self.maxiter = maxiter
        
        if twostep:
            self.betas = self.heckit()
        else:            
            self.betas = None #ML!
        self._cache = {}

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
    def MoransI(self):
        if 'MoransI' not in self._cache:
            if self.w:
                sig2ia = float(self.betas[-1]**2)*(self.lambd*(np.dot(self.x2,self.coef_x2)+self.lambd))
                self.sig2i = self.sigma**2 - sig2ia
                self._cache['MoransI'] = moran_KP(self.w, self.u, self.sig2i)
            else:
                raise Exception, "W matrix not provided to calculate spatial test."
        return self._cache['MoransI']

    def heckit(self):
        step1 = probit((self.y1!=0).astype(int), self.x2, constant=False, maxiter=self.maxiter)
        Z = -np.dot(self.x2,step1.betas)
        self.lambd = norm.pdf(Z)/norm.cdf(-Z)
        lambdf = self.lambd[self.y1!=0]
        n1 = lambdf.shape[0]
        lambdf = lambdf.reshape(n1,1)        
        for k in range(self.k1):
            if k==0:
                x1f = (self.x1[:,k].reshape(self.n,1))[self.y1!=0].reshape(n1,1)
            else:
                x1f = np.hstack((x1f,(self.x1[:,k].reshape(self.n,1))[self.y1!=0].reshape(n1,1)))

        r = np.hstack((x1f,lambdf))
        step2 = BaseOLS(self.y1[self.y1!=0].reshape(n1,1),r,constant=False)
        if self.twostep: #Two-step Heckman (VC not working)            
            Zf = Z[self.y1!=0].reshape(n1,1)
            dLdZ = lambdf*(lambdf - Zf)
            sig_0 = (step2.betas[-1]**2)*sum(-dLdZ)
            sigma2 = (np.dot(step2.u.T,step2.u) - sig_0)/n1
            self.sigma = np.sqrt(float(sigma2))
            self.rho = abs(float(step2.betas[-1]/self.sigma))                        
            for k in range(self.k2):
                if k==0:
                    x2f = (self.x2[:,k].reshape(self.n,1))[self.y1!=0].reshape(n1,1)
                else:
                    x2f = np.hstack((x2f,(self.x2[:,k].reshape(self.n,1))[self.y1!=0].reshape(n1,1)))
            B = la.inv(np.dot(r.T,r))
            """
            #VM as in Greene(1981)(slower):
            eta = 1 -dLdZ*(self.rho**2)
            W1 = step2.betas[-1]*np.sqrt(n1/self.n)*n1*dLdZ*r
            F = np.dot(W1.T,x2f)
            FEF = np.dot(np.dot(F,step1.vm),F.T)
            rs = np.sqrt(eta)*r*self.sigma
            psi1 = np.dot(rs.T,rs)            
            self.vm = np.dot(B,np.dot((psi1+FEF),B))
            """
            #VM as in Greene(2005):
            F2 = np.dot(r.T*dLdZ.T,x2f)            
            Q = (self.rho**2)*np.dot(F2,np.dot(step1.vm,F2.T))
            self.vm = sigma2*np.dot(np.dot(B,(np.dot(r.T,(1-((self.rho**2)*dLdZ))*r)+Q)),B)
            self.coef_x2 = step1.betas            
            self.vm_x2 = step1.vm
            self.n1 = n1
            self.predy = np.dot(np.hstack((self.x1,self.lambd)),step2.betas) #Matches Stata ycond, not xb (Stata's default).
            self.u = self.y1 - self.predy
        return step2.betas

#"""
def _test():
    import doctest
    doctest.testmod()

if __name__ == '__main__':
    #_test()
    import numpy as np
    import pysal
    """
    n = 100
    np.random.seed(5)
    x1 = np.random.uniform(0,10,(n,1))
    x1 = np.hstack((np.ones((n,1)),x1))
    x1b = np.dot(x1,np.reshape(np.array([1,0.5]),(2,1)))
    x2 = np.hstack((x1,np.random.uniform(14,0,(n,1))))
    x2b = np.dot(x2,np.reshape(np.array([1,0.5,-0.5]),(3,1)))    
    w = pysal.lat2W(int(np.sqrt(n)),int(np.sqrt(n)))
    w.transform='r'
    e2 = np.random.normal(0,1,(n,1))
    y2 = x2b + e2
    y1s = x1b + 0.8660254*np.random.normal(0,1,(n,1)) + 0.5*e2
    y1 = y1s*((y2>0).astype(int)) 
    heck1=heckman(y1,x1,x2,constant1=False,constant2=False,twostep=True,w=w)#,optim='ncg'
    """
    db = pysal.open("examples/mroz87.csv",'r')
    var = {}
    var['y']="wage"
    var['x1']=("exper","educ","city")
    var['x2']=("age","faminc","kids5","educ")
    y1 = np.array(db.by_col(var['y']))
    y1 = np.reshape(y1, (y1.shape[0],1))
    x1 = []
    for i in var['x1']:
        x1.append(db.by_col(i))

    x1 = np.array(x1).T
    x2 = []
    for i in var['x2']:
        x2.append(db.by_col(i))

    x2 = np.array(x2).T
    w = pysal.lat2W(3,251)
    w.transform='r'    
    heck1=heckman(y1,x1,x2,twostep=True,w=w)
    #"""
    print "Variable  Coef.  S.E.  Z-Stat. p-Value"
    print "Constant %5.4f %5.4f %5.4f %5.4f" % (heck1.betas[0],np.sqrt(heck1.vm.diagonal())[0],heck1.Zstat[0][0],heck1.Zstat[0][1])
    for i in range(x1.shape[1]):
        print "%-9s %5.4f %5.4f %5.4f %5.4f" % (var['x1'][i],heck1.betas[i+1],np.sqrt(heck1.vm.diagonal())[i+1],heck1.Zstat[i+1][0],heck1.Zstat[i+1][1])
    print "Lambda    %5.4f %5.4f %5.4f %5.4f" % (heck1.betas[-1],np.sqrt(heck1.vm.diagonal())[-1],heck1.Zstat[-1][0],heck1.Zstat[-1][1])
    print "Moran's I:", round(heck1.MoransI[0],3), "; pvalue:", round(heck1.MoransI[1],4)
    print "Sigma:", float(heck1.sigma)
    print "Rho:", heck1.rho
    
