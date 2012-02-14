import pysal, struct, os, time, sys, pickle
import numpy as np
import numpy.linalg as la
import scipy.sparse as SP
import scipy.stats as stats
import scipy.linalg as sla
import multiprocessing as mp
import econometrics.probit as pb
import scipy.optimize as op
from pb_sp_gridworker import get_grid, run_grid, grid_results

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

    References
    ----------
    .. [1] 
    
    Examples
    --------
    """

    def __init__(self,y,x,w,constant=True,R=None,core='single',verbose=False,lambda0=None,path='misc',ident=None):
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
        self.R = int(R)
        self.path = path
        self.core = core
        self.ident = ident        
        pars = self.par_est(lambda0)
        self.par = pars[0]
        self.logl = -pars[1] - np.log(R)
        self.vm = pars[3]
        self._cache = {}

    def par_est(self,lambda0,cores=None,pool=None):
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
            cores = int(cores)
        p = lambda par: self.get_p(par,I,Z,z,w_wp,wwp,self.R,cores,pool)        
        self.pb0 = pb.probit(self.y,self.x,constant=False,w=self.w) #Classic probit
        self.start = check_start(lambda0)
        self.p_res = {}
        for start_i in self.start:
            start_i = [start_i]
            for i in range(self.pb0.betas.shape[0]):
                start_i.append(float(self.pb0.betas[i]))
            if self.verbose:
                print 'Start:', start_i
            par_hat = op.fmin_bfgs(p,start_i,full_output=1)
            print '############### par_hat1:',par_hat
            self.p_res[par_hat[1]] = par_hat
            outfile = 'logl%s_%s.pkl' %(self.ident,int(start_i[0]*100))
            output = open(outfile, 'wb')
            pickle.dump(par_hat, output, -1)
            output.close()

        par_hat2 = self.p_res[min(self.p_res.keys())]
        print '############### par_hat2:',par_hat2
        counter=0
        while max(par_hat2[3].diagonal()) and min(par_hat2[3].diagonal())==1. and counter<15:
            if self.verbose:
                print 'Optimization failed. Re-starting...'
            par_hat2 = op.fmin_bfgs(p,par_hat2[0],full_output=1)
            print 'par_hat_i:',counter,par_hat2
            counter += 1
        return par_hat2

    def get_p(self,par,I,Z,z,w_wp,wwp,R,cores,pool):
        '''
        Builds the V and B matrices and calls the function to evaluate the boundaries.
        '''
        lambd = float(par[0])
        if abs(lambd)>=1.:
            if self.verbose:
                print lambd
                print "Lambda out of bounds. Artificial value assigned to ln(p)."        
            return 1e+300
        beta = []
        for i in range(self.k):
            beta.append(float(par[i+1]))
        beta = np.reshape(np.array(beta),(self.k,1))
        V = -z*np.dot(self.x,beta)
        if self.verbose:
            print 'lambd:', lambd, 'betas:', beta
        t1 = time.time()
        if lambd == 0:
            B = np.eye(self.n)
        else:
            try:
                a0 = I - lambd*w_wp + lambd*lambd*wwp
                a = Z*a0*Z.T
                if self.verbose:
                    print "Starting Cholesky, time elapsed to get here:", t1 - self.t0
                A = la.cholesky(a.todense()) #Cholesky decomposition on 'de-sparsified' matrix
            except:
                if self.verbose:
                    print 'Cholesky failed. Assigning artificial value.'
                return 1e+300
            if self.verbose:
                t2 = time.time()
                print "Cholesky finished. Starting inverse, time elapsed:", t2 - t1
            invt, = sla.lapack.get_lapack_funcs(('trtri',))
            B = invt(A.T)[0]
            if self.verbose:
                t3 = time.time()
                print "Inverse finished. Starting runs to find p, time elapsed:", t3 - t2
        sump = []
        if self.core == 'single':
            sump = p_runs([R,self.n,V,B,self.verbose])
        if self.core == 'multi':
            map(sump.extend,pool.map(p_runs, [(R/cores,self.n,V,B,self.verbose)] * cores))
        if self.core == 'grid':
            cores = min(max(R*225/self.n,25),R/10)
            IDs = range(R/cores)
            sump = get_grid((self.n,V,B,self.path),cores,IDs)
        if self.core == 'multi' or self.core == 'grid':
            if (R/cores)*cores < R:
                sump.extend(p_runs([R-int(R/cores)*cores,self.n,V,B,self.verbose]))
        if sump.count('Fail')>0:
            if self.verbose:
                print "Artificial value assigned to ln(p)."
            return 1e+300
        sump = np.array(sump)
        minsump = np.min(sump)
        sump = sump - minsump
        lnp = np.log(np.sum(np.exp(sump))) + minsump
        if self.verbose:
            print 'ln(p) =', lnp
        return -lnp

def p_runs(att):
    """
    Evaluates the boundaries and gets the sum over R of the product of the cumulative distributions.
    """
    R, N, V, B, verbose = att
    sumPhi = []
    seed = abs(struct.unpack('i',os.urandom(4))[0])
    np.random.seed(seed)
    for r in range(R):
        nn = np.zeros((N,1),float)
        vn = np.zeros((N,1),float)        
        sumbn = 0.
        prodPhi = 0.
        for i in range(N):
            n = -(i+1)
            vn[n] = 1.*(V[n]-sumbn)/B[n,n]
            prodPhi += np.log(stats.norm.cdf(vn[n]))
            if i<N-1:
                nn[n] = np.random.normal(0,1)
                tdraw = time.time()
                while nn[n] >= vn[n]:
                    nn[n] = np.random.normal(0,1)
                    if time.time() - tdraw > 15:
                        if verbose:
                            print '### Time limit reached. Artificial value assigned to ln(p). ###'
                            print 'Failed boundary:', vn[n] 
                        return ['Fail']
                sumbn = np.dot(B[n-1:n,n:],nn[n:])
        sumPhi.append(float(prodPhi))
    return sumPhi

def check_start(lambda0):
    if lambda0 == None:
        start = [0.0]
    elif issubclass(type(lambda0), float):
        start = [lambda0]
    elif issubclass(type(lambda0), list):
        start = lambda0        
    else:
        raise Exception, "invalid value passed to lambda0"
    return start

if __name__ == '__main__':
    #_test()
    import econometrics.power_expansion as PE
    lattice = 15
    n = lattice*lattice
    np.random.seed(100)
    x = np.random.uniform(-7,3,(n,1))     
    x = np.hstack((np.ones(x.shape),x))    
    w = pysal.lat2W(lattice,lattice)
    w.transform='r'
    b = np.reshape(np.array([1,0.5]),(2,1))    
    u = np.random.normal(0,1,(n,1))
    y = np.dot(x,b) + PE.power_expansion(w, u, 0.5) #Build y{star}

    probit1=probit_sp((y>0).astype(float),x,w,core='multi',R=500,constant=False,verbose=True)
    print "Total time elapsed:", time.time() - probit1.t0
    print "Parameters (lambda, beta_0, beta_1) =", probit1.par
    print "Variance-covariance=", probit1.vm
    print "Log likelihood=", probit1.logl
    print "Number of runs performed:", probit1.R,", N =", probit1.n
    print 'Start:', probit1.start
