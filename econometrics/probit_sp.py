import pysal, struct, os, pickle, glob, time
import numpy as np
import numpy.linalg as la
import scipy.sparse as SP
import scipy.stats as stats
import scipy.linalg as sla
import multiprocessing as mp
import probit as pb
import scipy.optimize as op
import grid_loader as gl
#import scikits.sparse.cholmod as CM

class probit_sp:
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
                  Default: min(sqrt(n),100)(?????)
    core        : string
                  How the computer's cores will be used to run the simulations.
                  Default: 'single' (Single core)
                  Alternative: 'multi' (Multiple cores of the local machine)
              
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
                  'single','multi'
    scale       : integer
                  How many times the likelihood was re-scaled (multiplied) by 1e+200.

    References
    ----------
    .. [1] 
    
    Examples
    --------
    """

    def __init__(self,y,x,w,constant=True,R=None,core='single',verbose=False,lambda0=None):
        self.t0 = time.time()
        self.y = y        
        if constant:
            x = np.hstack((np.ones(y.shape),x))
        self.constant = constant
        self.x = x
        self.n, self.k = x.shape
        self.w = w
        self.verbose = verbose
        if R==None:  #R = Number of runs for p
            R = np.minimum(int(np.sqrt(self.n)),100)
        self.R = R
        self.core = core        
        pars = self.par_est(lambda0)
        self.par = pars
        self.logl = -pars[1]
        self._cache = {}

    def par_est(self,lambda0):
        '''
        Sets up matrices and start values to begin the simulations and call the functions to get the parameter estimates.
        '''
        I = SP.lil_matrix(self.w.sparse.get_shape()) #Set sparse identity matrix
        I.setdiag(np.ones((self.n,1),float).flat)
        I = I.asformat('csr')
        w_wp = self.w.sparse + self.w.sparse.T
        wwp = self.w.sparse.T*self.w.sparse
        z = 1-2*self.y
        Z = SP.lil_matrix(self.w.sparse.get_shape())
        Z.setdiag(z.flat)
        Z = Z.asformat('csr')
        if self.core == 'multi':
            cores = mp.cpu_count()
            pool = mp.Pool(cores)
            cores = int(cores/2) #Using only half of the cores available here since the X Grid is using the others.
            p = lambda par: self.get_p(par,I,Z,z,w_wp,wwp,self.R,cores=cores,pool=pool)
        else:
            p = lambda par: self.get_p(par,I,Z,z,w_wp,wwp,self.R)        
        pb0 = pb.probit(self.y,self.x,constant=False,w=self.w) #Classic probit
        if lambda0 == None:
            start = [0.0]
        else:
            start = [lambda0]
        bounds = [(-0.9999,0.9999)]
        for i in pb0.betas:
            start.append(i)
            bounds.append((None,None))
        self.scale = -1
        par_hat = op.fmin_l_bfgs_b(p,start,approx_grad=True,bounds=bounds)#,epsilon=0.0001,factr=1000000.0,
        self.pb0 = pb0
        return par_hat

    def get_p(self,par,I,Z,z,w_wp,wwp,R0,cores=None,pool=None):
        '''
        Builds the V and B matrices and calls the function to evaluate the boundaries.
        '''
        #t0 = time.time()
        beta = []
        for i in range(self.k+1)[1:]:
            beta.append(float(par[i]))         
        beta = np.reshape(np.array(beta),(self.k,1))
        lambd = float(par[0])
        V = -z*np.dot(self.x,beta)        
        if self.verbose:
            print 'lambd:', lambd, 'betas:', beta
        t1 = time.time()        
        #try:
        if lambd == 0:
            B = np.eye(self.n)
        else:
            a0 = I - lambd*w_wp + lambd*lambd*wwp
            a = np.dot(Z,np.dot(a0,Z.T))
            if self.verbose:
                print "Starting Cholesky, time elapsed to get here:", t1 - self.t0
            A = la.cholesky(a.todense()) #Cholesky decomposition on 'de-sparsified' matrix
            #A = CM.cholesky(a) #Possible sparse solution, but has license issues
            if self.verbose:
                t2 = time.time()
                print "Cholesky finished. Starting inverse, time elapsed:", t2 - t1
            B = la.inv(A.T)
            if self.verbose:
                t3 = time.time()
                print "Inverse finished. Starting runs to find p, time elapsed:", t3 - t2
        if self.scale < 0:
            sump, self.scale = p_runs([1,self.n,V,B,self.verbose,self.scale])
            R = R0-1
            if self.verbose:
                print 'Scale:', self.scale
        else:
            R = R0
            sump = 0.
        if self.core == 'single':
            sump += p_runs([R,self.n,V,B,self.verbose,self.scale])
        if self.core == 'multi':
            sump += sum(pool.map(p_runs, [(R/cores,self.n,V,B,self.verbose,self.scale)] * cores))         
        if self.core == 'multi' or self.core == 'grid':
            if int(R/cores)*cores < R:
                sump += p_runs([R-int(R/cores)*cores,self.n,V,B,self.verbose,self.scale])
        lnp = np.log(1.0*sump/R0)
        print 'ln(p) =', lnp
        #except:
        return -lnp

def p_runs(att):
    """
    Evaluates the boundaries and gets the sum over R of the product of the cumulative distributions.
    """
    R, N, V, B, verbose, scale1 = att
    sumPhi = 0
    if scale1 < 0:
        scale0 = 0
    else:
        scale0 = scale1
    for r in range(R):
        seed = abs(struct.unpack('i',os.urandom(4))[0])
        np.random.seed(seed)
        nn = np.zeros((N,1),float)
        vn = np.zeros((N,1),float)        
        sumbn = 0
        prodPhi = 1.0
        for i in range(N):
            n = -(i+1)
            vn[n] = 1.0*(V[n]-sumbn)/B[n,n]
            prodPhi = prodPhi * stats.norm.cdf(vn[n])  #* np.exp(1)
            if prodPhi < 1e-50 and scale1 < 0 and r == 0:
                prodPhi = prodPhi * 1e+200
                scale0 += 1
            if prodPhi < 1e-50 and scale1 > 0:
                prodPhi = prodPhi * 1e+200
                scale1 -= 1
            if i<N-1:
                nn[n] = np.random.normal(0,1)                
                tdraw = time.time()
                while nn[n] >= vn[n]:
                    nn[n] = np.random.normal(0,1)
                    if time.time() - tdraw > 15:
                        sumPhi = 1e-320
                        if verbose:
                            print '### Time limit reached. Artificial value assigned to ln(p). ###'
                            print 'Failed boundary:', vn[n] 
                        return sumPhi
                sumbn = np.dot(B[n-1:n,n:],nn[n:])
        if scale1 > 0:
            for i in range(scale1):
                prodPhi = prodPhi * 1e+200
        sumPhi += prodPhi
    if scale1 < 0:
        return float(sumPhi), scale0
    else:
        return float(sumPhi)


if __name__ == '__main__':
    #_test()
    import power_expansion as PE
    lattice = 30
    n = lattice*lattice
    x = np.random.uniform(-4,0,(n,1))     
    x = np.hstack((np.ones(x.shape),x))    
    w = pysal.lat2W(lattice,lattice)
    w.transform='r'
    b = np.reshape(np.array([1,0.5]),(2,1))    
    u = np.random.normal(0,1,(n,1))
    ys = np.dot(x,b) + PE.power_expansion(w, u, 0.3) #Build y{star}
    y = np.zeros(ys.shape,float) #Binary y
    for i in range(len(y)):
        if ys[i]>0:
            y[i] = 1

    probit1=probit_sp(y,x,w,core='single',R=50,constant=False,verbose=True)
    print "Total time elapsed:", time.time() - probit1.t0
    print "Parameters (lambda, beta_0, beta_1) =", probit1.par
    print "Log likelihood=", probit1.logl
    print "Number of runs performed:", probit1.R,", N =", probit1.n
    print 'Scale:', probit1.scale
