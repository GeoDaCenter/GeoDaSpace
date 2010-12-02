import pysal
import numpy as np
import numpy.linalg as la
import scipy.sparse as SP
import scipy.stats as stats
import multiprocessing as mp
import struct
import os
#import scipy.optimize as op
#import scikits.sparse.cholmod as CM
import time

class probit_sp: #DEV class required.
    """
    Spatial probit class to do all the computations

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
    R           : integer
                  Number of runs to be performed
                  Default: 5*sqrt(n)(?????)
    core        : string
                  How the computer's cores will be used to run the simulations.
                  Default: 'single' (Single core)
                  Alternative: 'multi' (Multiple cores of the local machine)
                               'grid' (X Grid - not yet implemented)
              
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
    R           : integer
                  Number of runs performed
    core        : string
                  How the computer's cores will be used to run the simulations.
                  'single','multi' or 'grid'

    References
    ----------
    .. [1] 
    
    Examples
    --------
    """
    def __init__(self,x,y,w,constant=True,R=None,core='single'):
        self.t0 = time.time()
        self.y = y        
        if constant:
            x = np.hstack((np.ones(y.shape),x))
        self.constant = constant
        self.x = x
        self.n, self.k = x.shape
        self.w = w
        if R==None:
            R = 5*int(np.sqrt(self.n))
        self.R = R
        self.core = core
        par_est = self.par_est()
        self._cache = {}

    def par_est(self):
        '''
        Sets up matrices and start values to begin the simulations and call the functions to get the parameter estimates.
        '''
        I = SP.lil_matrix(w.sparse.get_shape()) #Set sparse identity matrix
        I.setdiag(np.ones((self.n,1),float).flat)
        I = I.asformat('csr')
        w_wp = w.sparse + w.sparse.T
        wwp = w.sparse*w.sparse.T
        start = np.vstack((np.dot(la.inv(np.dot(self.x.T,self.x)),np.dot(self.x.T,self.y)),0))
        if self.core == 'multi':
            cores = mp.cpu_count()
            pool = mp.Pool(cores)
            cores = int(cores/2) #Using only half of the cores available here since the X Grid is using the others.
            p = self.get_p(start,I,w_wp,wwp,self.R,cores=cores,pool=pool)
            #p = lambda par: self.get_p(par,I,w_wp,wwp,self.R,cores=cores,pool=pool) #lambda functions should be used if scipy optimizers are to be used.
        if self.core == 'single':
            p = self.get_p(start,I,w_wp,wwp,self.R)
            #p = lambda par: self.get_p(par,I,w_wp,wwp,self.R)            
        t1 = time.time()
        print "Total time elapsed:", t1-self.t0
        print "ln(p) =", p
        print "Number of runs performed:", self.R
        #par_hat = op.fmin(p,start)        

    def get_p(self,par,I,w_wp,wwp,R,cores=None,pool=None): #R = Number of runs for p
        '''
        Builds the V and B matrices and calls the function to evaluate the boundaries.
        '''
        beta = [] #Includes lambda
        for i in range(self.k):
            beta.append(float(par[i]))         
        beta = np.reshape(np.array(beta[0:self.k]),(self.k,1))
        lambd = float(beta[-1])
        a = (I + lambd*w_wp + lambd*lambd*wwp)*np.eye(self.n) #Multiplied by I to 'de-sparsify'
        t1 = time.time()
        print "Starting Cholesky, time elapsed to get here:", t1-self.t0
        A = la.cholesky(a).T
        #A = CM.cholesky(I + lambd*w_wp + lambd*lambd*wwp) #Possible sparse solution
        #B = la.inv(A.L())
        t1 = time.time()
        print "Cholesky finished. Starting inverse, time elapsed:", t1-self.t0
        B = la.inv(A)
        t1 = time.time()
        print "Inverse finished. Starting runs to find p, time elapsed:", t1-self.t0
        V = np.dot(self.x,beta)*(2*self.y-1)
        if self.core == 'single':
            sigp = p_runs([R,self.n,V,B])
        if self.core == 'multi':
            sigp = sum(pool.map(p_runs, [(R/cores,self.n,V,B)] * cores))
            if int(R/cores)*cores < R:
                 sigp += p_runs([R-R/cores*cores,self.n,V,B])
        return np.log(float(1.0*sigp/R))

def p_runs(att):
    """
    Evaluates the boundaries and gets the sum over R of the product of the cumulative distributions.
    """
    R = att[0]
    N = att[1]
    V = att[2]
    B = att[3]
    sumPhi = 0
    for r in range(R):
        seed = abs(struct.unpack('i',os.urandom(4))[0])
        np.random.seed(seed)
        nn = np.zeros((N,1),float)
        vn = np.zeros((N,1),float)
        sumbn = 0
        prodPhi = 1
        for i in range(N):
            n = -(i+1)
            vn[n] = 1.0*(V[n]-sumbn)/B[n,n]
            prodPhi = prodPhi * stats.norm.cdf(vn[n])
            nn[n] = np.random.normal(0,1)
            while nn[n] >= vn[n]:
                nn[n] = np.random.normal(0,1)
            if i<N-1:
                sumbn = np.dot(B[n-1:n,n:],nn[n:])
        sumPhi += prodPhi
    return sumPhi

if __name__ == '__main__':
    #_test()
    lattice = 10
    n = lattice*lattice
    x = np.random.uniform(-4,0,(n,1))
    x = np.hstack((np.ones(x.shape),x))    
    w = pysal.lat2W(lattice,lattice)
    w.transform='r'
    b = np.reshape(np.array([1,0.5]),(2,1))
    e = np.dot(la.inv((np.eye(n)-0.8*w.full()[0])),np.random.normal(0,1,(n,1)))
    ys = np.dot(x,b) + e #Build y{star}
    y = np.zeros((n,1),float) #Binary y
    for yi in range(len(y)):
        if ys[yi]>0:
            y[yi] = 1
    probit1=probit_sp(x,y,w,core='multi',constant=False)
