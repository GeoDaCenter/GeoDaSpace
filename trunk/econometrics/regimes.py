import time
import numpy as np
import pandas as pd
import pysal as ps
import scipy.sparse as SP
import itertools as iter
# For OLS
from pysal.spreg.utils import RegressionPropsY, RegressionPropsVM
import pysal.spreg.robust as ROBUST
import numpy.linalg as la

class BaseOLS_sp(RegressionPropsY, RegressionPropsVM):
    def __init__(self, y, x, constant=True,\
                 robust=None, gwk=None, sig2n_k=True):

        if constant:
            self.x = SP.hstack((SP.csr_matrix(np.ones(y.shape)), x))
        else:
            self.x = x
        self.xtx = (self.x.T * self.x).toarray()
        self.xtxi = la.inv(self.xtx)
        xty = self.x.T * y
        self.betas = np.dot(self.xtxi, xty)
        predy = self.x * self.betas
        u = y-predy
        self.u = u
        self.predy = predy
        self.y = y
        self.n, self.k = self.x.shape

        if robust:
            self.vm = ROBUST.robust_vm(reg=self, gwk=gwk)

        self._cache = {}
        if sig2n_k:
            self.sig2 = self.sig2n_k
        else:
            self.sig2 = self.sig2n

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
                  The structure of the alignent is X1r1 X1r2 ... X2r1 X2r2 ...
    '''
    n, k = x.shape
    regimes_set = list(set(regimes))
    regimes_set.sort()
    data = x.flatten()
    R = len(regimes_set)
    regime_by_row = np.array(list(regimes_set) * k)
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
    n, k, r = (3000000, 9, 50)
    #n, k, r = (50000, 26, 359)
    #n, k, r = (10, 5, 3)
    print('Using setup with n=%i, k=%i and %i regimes'%(n, k, r))
    x = np.random.random((n, k))
    y = np.random.random((n, 1))
    inR1 = n / 2
    inR2 = n / 2
    regimes = ['r1'] * inR1 + ['r2'] * inR2
    nr = [int(np.round(n/r))] * (r-1)
    regimes = list(iter.chain(*[['r'+str(i)]*j for i, j in enumerate(nr)]))
    regimes = regimes + ['r'+str(r-1)] * (n-len(regimes)) 
    #np.random.shuffle(regimes)

    # Self-cooked option
    t0 = time.time()
    xc = np.hstack((np.ones(y.shape), x))
    xsp = x2xsp(xc, regimes)
    #xc = SP.hstack((SP.csr_matrix(np.random.random(x.shape)), xsp))
    t1 = time.time()
    print('XS_csr created in %f seconds'%(t1-t0))

    '''
    t0 = time.time()
    xc = np.hstack((np.ones(y.shape), x))
    xsp_csc = x2xsp_csc(xc, regimes)
    t1 = time.time()
    print('XS_csc created in %f seconds'%(t1-t0))

    print '\n'
    t0 = time.time()
    ols_sp = BaseOLS_sp(y, xsp, constant=False)
    t1 = time.time()
    print('OLS run in %f seconds'%(t1-t0))
    ols1 = ps.spreg.ols.BaseOLS(y[:inR1, :], x[:inR1, :])
    ols2 = ps.spreg.ols.BaseOLS(y[inR1:, :], x[inR1:, :])
    print 'Regime 1 (pooled, independent):'
    print np.hstack((ols_sp.betas[range(0, 8, 2)], ols1.betas))
    print ''
    print 'Regime 2 (pooled, independent):'
    print np.hstack((ols_sp.betas[range(1, 9, 2)], ols2.betas))
    '''
