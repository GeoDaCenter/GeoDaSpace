import time
import numpy as np
import pandas as pd
import pysal as ps
import scipy.sparse as SP
import itertools as iter
from scipy.stats import f
# For OLS
from pysal.spreg.utils import RegressionPropsY, RegressionPropsVM
import pysal.spreg.robust as ROBUST
import numpy.linalg as la

class BaseOLS_sp(RegressionPropsY, RegressionPropsVM):
    def __init__(self, y, xsp, constant=True,\
                 robust=None, gwk=None, sig2n_k=True):

        if constant:
            self.xsp = SP.hstack((SP.csr_matrix(np.ones(y.shape)), xsp))
        else:
            self.xsp = xsp
        self.xtx = (self.xsp.T * self.xsp).toarray()
        self.xtxi = la.inv(self.xtx)
        xty = self.xsp.T * y
        self.betas = np.dot(self.xtxi, xty)
        predy = self.xsp * self.betas
        u = y-predy
        self.u = u
        self.predy = predy
        self.y = y
        self.n, self.k = self.xsp.shape

        if robust:
            self.vm = ROBUST.robust_vm(reg=self, gwk=gwk)

        self._cache = {}
        if sig2n_k:
            self.sig2 = self.sig2n_k
        else:
            self.sig2 = self.sig2n

class Chow:
    '''
    Traditional Chow test for parameter instability. Implemented following
    first equation of page 192 of Anselin (1990) [1]_
    ...

    Arguments
    =========

    ols_sp  : OLS_sp
              Sparse OLS regression object 
    x       : array
              Original nxk dense array with observations and variables used in
              the regime implementation of ols_sp

    Attributes
    ==========

    chow    : float
              Value of the Chow statistic
    p       : float
              P-value associated to the Chow statistic, distributed as a F
              with K and (N - 2K) degrees of freedom

    References
    ==========

    .. [1] Anselin, L. (1990) "Spatial Dependence and Spatial Structural
    Instability in Applied Regression Analysis" Journal of Regional Science,
    vol. 30, No. 2, pp. 185-207

    '''
    def __init__(self, ols_sp, x):
        n, k = x.shape
        if ols_sp.xsp.constant:
            constant=True
        else:
            constant=False
        ols = ps.spreg.ols.BaseOLS(ols_sp.y, x, constant=constant)
        u_r = ols.u
        u_u = ols_sp.u
        chow, p = chow_test(u_u, u_r, n, k)
        self.chow = chow
        self.p = p

def chow_test(u_u, u_r, n, k):
    '''
    Stripped down Chow test implemented following first equation of page 192
    of Anselin (1990) [1]_
    ...

    Arguments
    =========

    u_u     : array
              Vector of dimension nx1 with residuals from unrestricted
              model
    u_r     : array
              Vector of dimension nx1 with residuals from restricted
              model
    n       : int
              Number of observations
    k       : int
              Number of variables

    Returns
    =======

    chow    : float
              Value of the Chow statistic
    p       : float
              P-value associated to the Chow statistic, distributed as a F
              with K and (N - 2K) degrees of freedom

    References
    ==========

    .. [1] Anselin, L. (1990) "Spatial Dependence and Spatial Structural
    Instability in Applied Regression Analysis" Journal of Regional Science,
    vol. 30, No. 2, pp. 185-207

    '''
    n, k = map(float, [n, k])
    uutuu = np.dot(u_u.T, u_u)
    nm2k = n - 2.*k
    num = (np.dot(u_r.T, u_r) - uutuu) / float(k)
    den = uutuu / nm2k
    chow = num / den
    p = f.sf(chow, k, nm2k)
    return chow, p

    fStat = (U/(k-1))/(Q/(n-k))
    pValue = stats.f.sf(fStat,k-1,n-k)
    fs_result = (fStat, pValue)

def regimeX_setup(x, regimes, cols2regi, constant=False):
    '''
    Flexible full setup of a regime structure
    ...

    Arguments
    =========
    x           : np.array
                  Dense array of dimension (n, k) with values for all observations
    regimes     : list
                  list of n values with the mapping of each observation to a
                  regime. Assumed to be aligned with 'x'.
    cols2regi   : list
                  List of k booleans indicating whether each column should be
                  considered as different per regime (True) or held constant
                  across regimes (False)
    constant    : [False, 'one', 'many']
                  Switcher controlling the constant term setup. It may take
                  the following values:
                    
                    *  False: no constant term is appended in any way
                    *  'one': a vector of ones is appended to x and held
                              constant across regimes
                    * 'many': a vector of ones is appended to x and considered
                              different per regime

    Returns
    =======
    xsp         : csr sparse matrix
                  Sparse matrix containing the full setup for a regimes model
                  as specified in the arguments passed
                  NOTE: columns are reordered so first are all the regime
                  columns then all the global columns (this makes it much more
                  efficient)
                  Structure of the output matrix (assuming X1, X2 to vary
                  across regimes and constant term, X3 and X4 to be global):
                    X1r1, X2r1, ... , X1r2, X2r2, ... , constant, X3, X4
                  Besides the normal attributes of a csr sparse matrix, xsp
                  has the following appended as meta-data:
                    
                    * regimes
                    * cols2regi
                    * constant
    '''
    n, k = x.shape
    if constant:
        x = np.hstack((np.ones((n, 1)), x))
        if constant == 'one':
            cols2regi.insert(0, False)
        elif constant == 'many':
            cols2regi.insert(0, True)
        else:
            raise Exception, "Invalid argument (%s) passed for 'constant'. Please secify a valid term."%str(constant)
    cols2regi = np.array(cols2regi)
    if len(set(cols2regi))==1:
        xsp = x2xsp(x, regimes)
    else:
        not_regi = x[:, np.where(cols2regi==False)[0]]
        regi_subset = x[:, np.where(cols2regi)[0]]
        regi_subset = x2xsp(regi_subset, regimes)
        xsp = SP.hstack( (regi_subset, SP.csr_matrix(not_regi)) )
    xsp.regimes = regimes
    xsp.cols2regi = cols2regi
    xsp.constant = constant
    return xsp

def x2xsp(x, regimes):
    '''
    Convert X matrix with regimes into a sparse X matrix that accounts for the
    regimes
    ...

    Attributes
    ==========
    x           : np.array
                  Dense array of dimension (n, k) with values for all observations
    regimes     : list
                  list of n values with the mapping of each observation to a
                  regime. Assumed to be aligned with 'x'.
    Returns
    =======
    xsp         : csr sparse matrix
                  Sparse matrix containing X variables properly aligned for
                  regimes regression. 'xsp' is of dimension (n, k*r) where 'r'
                  is the number of different regimes
                  The structure of the alignent is X1r1 X2r1 ... X1r2 X2r2 ...
    '''
    n, k = x.shape
    regimes_set = list(set(regimes))
    regimes_set.sort()
    data = x.flatten()
    R = len(regimes_set)
    regime_by_row = np.array([[r] * k for r in list(regimes_set)]).flatten() #X1r1 X2r1 ... X1r2 X2r2 ...
    row_map = {r: np.where(regime_by_row == r)[0] for r in regimes_set}
    indices = np.array([row_map[row] for row in regimes]).flatten()
    indptr = np.zeros((n+1, ), dtype=int)
    indptr[:-1] = list(np.arange(n) * k)
    indptr[-1] = n*k
    return SP.csr_matrix((data, indices, indptr))

def x2xsp_csc(x, regimes):
    '''
    Implementation of x2xsp based on csc sparse matrices

    Slower as r grows

    NOTE: for legacy purposes
    '''
    n, k = x.shape
    regimes_set = list(set(regimes))
    regimes_set.sort()
    regimes = np.array(regimes)
    data = x.flatten('F')
    R = len(regimes_set)
    col_map = {r: np.where(regimes == regimes_set[r])[0] for r in np.arange(R)}
    reg_order = np.array([np.arange(R) for i in np.arange(k)]).flatten()
    indices = list(iter.chain(*[col_map[r] for r in reg_order]))
    indptr = np.zeros(reg_order.shape[0] + 1, dtype=int)
    for i in np.arange(1, indptr.shape[0]):
        indptr[i] = indptr[i-1] + len(col_map[reg_order[i-1]])
    return SP.csc_matrix((data, indices, indptr)).tocsr()
    #return data, indices, indptr

def x2xsp_pandas(x, regimes):
    '''
    Similar functionality as x2xsp but using pandas as the core engine for
    realigning. Two main cons:
        * You have to build full XS before making it sparse
        * You have to convert XS DataFrame to an np.array before going to csr
    These make it slower and less memory efficient

    NOTE: for legacy purposes
    '''
    n, k = x.shape
    multiID = pd.MultiIndex.from_tuples(zip(np.arange(n), regimes), \
            names=['id', 'regime'])
    df = pd.DataFrame(x, index=multiID, columns=['x'+str(i) for \
            i in range(k)])
    a = df.unstack().fillna(0).as_matrix()
    return SP.csr_matrix(a)

if __name__ == '__main__':
    # Data Setup
    #n, kr, kf, r, constant = (1500000, 8, 1, 50, 'many')
    #n, kr, kf, r, constant = (50000, 16, 15, 359, 'many')
    #n, kr, kf, r = (50000, 11, 0, 359)
    n, kr, kf, r, constant = (10, 2, 0, 2, 'many')
    print('Using setup with n=%i, kr=%i, kf=%i, %i regimes and %s constant' \
            %(n, kr, kf, r, constant))
    k = kr + kf
    x = np.random.random((n, k))
    x[:50, :] = x[:50, :] * 100
    #x[:50, :] = np.random.random((50, k))
    y = np.random.random((n, 1))
    inR1 = n / 2
    inR2 = n / 2
    regimes = ['r1'] * inR1 + ['r2'] * inR2
    nr = [int(np.round(n/r))] * (r-1)
    regimes = list(iter.chain(*[['r'+str(i)]*j for i, j in enumerate(nr)]))
    regimes = regimes + ['r'+str(r-1)] * (n-len(regimes)) 
    #np.random.shuffle(regimes)

    # Regime model setup
    t0 = time.time()
    cols2regi = [True] * kr + [False] * kf
    xsp = regimeX_setup(x, regimes, cols2regi, constant=constant)
    t1 = time.time()
    print('Full setup created in %.4f seconds'%(t1-t0))

    '''
    # Regression and Chow test
    ols = BaseOLS_sp(y, xsp, constant=False)
    t2 = time.time()
    print('OLS run in %.4f seconds'%(t2-t1))
    chow = Chow(ols, x)
    print 'Chow test:  %.4f\t\tp-value:  %.4f'%(chow.chow, chow.p)
    '''

