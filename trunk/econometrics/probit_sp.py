import pysal
import numpy as np
import numpy.linalg as la
import scipy.sparse as SP
import scipy.stats as stats
import scipy.linalg as sla
import multiprocessing as mp
import struct
import os
import probit as pb
import scipy.optimize as op
import pickle
import glob
import grid_loader as gl
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
                               'grid' (X Grid)
              
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
            R = int(np.sqrt(self.n))
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
        #start = np.vstack((np.dot(la.inv(np.dot(self.x.T,self.x)),np.dot(self.x.T,self.y)),0)) #OLS        
        if self.core == 'multi':
            cores = mp.cpu_count()
            pool = mp.Pool(cores)
            cores = int(cores/2) #Using only half of the cores available here since the X Grid is using the others.
            #p = self.get_p(start,I,w_wp,wwp,self.R,cores=cores,pool=pool)
            p = lambda par: -self.get_p(par,I,w_wp,wwp,self.R,cores=cores,pool=pool) #lambda functions should be used if scipy optimizers are to be used.
        else:
            #p = self.get_p(start,I,w_wp,wwp,self.R)
            p = lambda par: -self.get_p(par,I,w_wp,wwp,self.R)        
        pb0 = pb.probit(x,y,constant=False).betas #Probit
        start = [0.0]
        for i in pb0:
            start.append(i)
        #self.par_hat = op.fmin_tnc(p,start)
        self.par_hat = op.fmin_l_bfgs_b(p,start,approx_grad=True,epsilon=0.0001,bounds=[(-0.9999,0.9999),(None,None),(None,None)])

    def get_p(self,par,I,w_wp,wwp,R,cores=None,pool=None): #R = Number of runs for p
        '''
        Builds the V and B matrices and calls the function to evaluate the boundaries.
        '''
        beta = [] #Includes lambda
        for i in range(self.k+1)[1:]:
            beta.append(float(par[i]))         
        beta = np.reshape(np.array(beta),(self.k,1))
        lambd = float(par[0])
        print lambd, beta[0], beta[1]
        V = np.dot(self.x,beta)*(2*self.y-1)        
        t1 = time.time()        
        #try:
        if t1:
            if lambd == 0:
                B = np.eye(self.n)
            else:
                a = I + lambd*w_wp + lambd*lambd*wwp
                #print "Starting Cholesky, time elapsed to get here:", t1-self.t0
                A = la.cholesky(a.todense()).T #Cholesky decomposition on 'de-sparsified' matrix             
                #A = CM.cholesky(a) #Possible sparse solution, but has license issues
                #B = get_inv(A) #Change get_inv to sparse functions
                t2 = time.time()
                #print "Cholesky finished. Starting inverse, time elapsed:", t2 - t1
                B = la.inv(A)
                t3 = time.time()
                #print "Inverse finished. Starting runs to find p, time elapsed:", t3 - t2        
            if self.core == 'single':
                sigp = p_runs([R,self.n,V,B])
            if self.core == 'multi':
                sigp = sum(pool.map(p_runs, [(R/cores,self.n,V,B)] * cores))         
            if self.core == 'grid':
                cores = 100 #amount of runs each processor will get.
                IDs = range(R/cores)                                        
                sigp = get_grid((self.n,V,B),cores,IDs)
                print "p =", sigp, sigp/R
            if self.core == 'multi' or self.core == 'grid':
                if int(R/cores)*cores < R:
                    sigp += p_runs([R-int(R/cores)*cores,self.n,V,B])
            lnp = np.log(float(1.0*sigp/R))
            print 'lnp =', lnp
        #except:

        return lnp

def get_grid(data,cycles,IDs):
    #Full path must be the folder in which you want to save the grid temp files and the gridworker /
    #without the first '/'.
    fullpath = 'Users/pedroamaral/Documents/Academico/GeodaCenter/python/SVN/spreg/trunk/econometrics/misc/'
    path = fullpath.split('/')[-2]+'/'
    infile = open(path+'probit_sp.pkl', 'wb')
    pickle.dump(data, infile, -1)
    infile.close()
    xgrid_ids = run_grid(cycles,IDs,fullpath)    
    repeat = grid_results(xgrid_ids,IDs,path)
    #Check which ones are done
    while repeat:
        done = []
        fname = path+'run_*.pkl'
        for i in glob.glob(fname): 
            i = i.split('.')
            i = i[0].split('/')
            i = i[1].split('_')
            i = int(i[1])
            done.append(i)        
        for i in IDs: #Enumerate which were not done,
            if done.count(i)>0:
                repeat.remove(i)
        if repeat: #and re-send them.
            print 'Repeating:', repeat
            xgrid_ids = run_grid(cycles,repeat,fullpath)
            repeat = grid_results(xgrid_ids,IDs,path)           
    output = []
    for i in range(len(IDs)):
        outfile = path+'run_%s.pkl' %i
        pkl_file = open(outfile, 'rb')
        data = pickle.load(pkl_file)
        pkl_file.close()
        output.append(data[0])
        #print data
        os.remove(outfile)
    os.remove(path+'probit_sp.pkl')
    return sum(output)

def run_grid(cycles,IDs,fullpath): #Send jobs
    xgrid_ids = []
    for i in IDs:    
        path = fullpath.split('/')[-2]+'/'
        args = ' %s %s %s %s' %(cycles,i,fullpath,path)        
        jid = gl.load_python(path+'pb_sp_gridworker.py',args=args)
        print 'loaded', jid, i
        xgrid_ids.append([jid, i])
    print 'all files loaded\n'
    return xgrid_ids

def grid_results(xgrid_ids,IDs,path):
    xgrid_ids = gl.results_runner(xgrid_ids,path,delay=5)
    repeat = []
    for i in IDs:
        repeat.append(i)
    time.sleep(5)
    return repeat

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
        prodPhi = 1.0
        for i in range(N):
            n = -(i+1)
            vn[n] = 1.0*(V[n]-sumbn)/B[n,n]
            prodPhi = prodPhi * stats.norm.cdf(vn[n])
            if i<N-1:
                nn[n] = np.random.normal(0,1)
                while nn[n] >= vn[n]:
                    nn[n] = np.random.normal(0,1)
                sumbn = np.dot(B[n-1:n,n:],nn[n:])
        sumPhi += prodPhi
    return sumPhi

def get_inv(A):
    lu = sla.lu_factor(A)
    Ai = []
    for i in range(A.shape[0]):
        b = np.zeros((A.shape[0],1))
        b[i] = 1
        Ai.append(sla.lu_solve(lu,b))
    return np.hstack((i for i in Ai))


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
    ys = np.dot(x,b) + PE.power_expansion(w, u, 0.5) #Build y{star}
    y = np.zeros(ys.shape,float) #Binary y
    for i in range(len(y)):
        if ys[i]>0:
            y[i] = 1
    #w.transform='b'
    probit1=probit_sp(x,y,w,core='grid',R=2000,constant=False)
    print "Total time elapsed:", time.time() - probit1.t0
    print "par_hat =", probit1.par_hat
    print "Number of runs performed:", probit1.R,", N =", probit1.n
