import time
import numpy as np
import pandas as pd
import pysal as ps
import scipy.sparse as SP
import itertools as iter
from scipy.stats import f, chisqprob
# For OLS
from pysal.spreg import diagnostics
from pysal.spreg.utils import RegressionPropsY, RegressionPropsVM
import pysal.spreg.robust as ROBUST
import numpy.linalg as la

class BaseOLS_sp(RegressionPropsY, RegressionPropsVM):
    def __init__(self, y, xsp, constant=True,\
                 robust=None, gwk=None, sig2n_k=True):

        if constant:
            self.xsp = SP.hstack((SP.csr_matrix(np.ones(y.shape)), xsp), format='csr')
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

class Chow_sp:
    '''
    Spatial Chow test of coefficient stability across regimes. The test is a
    particular case of the Wald statistic in which the constraint are setup
    according to the spatial regime structure
    ...

    Arguments
    =========
    reg     : regression object
              Regression object from PySAL.spreg which is assumed to have the
              following attributes:
                    
                    * betas : coefficient estimates
                    * vm    : variance covariance matrix of betas
                    * kr    : Number of variables varying across regimes
                    * kf    : Number of variables fixed (global) across regimes
                    * nr    : Number of regimes

    Attributes
    ==========
    joint   : tuple
              Pair of Wald statistic and p-value for the setup of global
              spatial stability, that is all betas are the same across
              regimes.
    regi    : array
              kr x 2 array with Wald statistic (col 0) and its p-value (col 1)
              for each beta that varies across regimes. The restrictions
              are setup to test for the global stability (all regimes have the
              same parameter) of the beta.
    '''
    def __init__(self, reg):
        kr, kf, nr, betas, vm = reg.kr, reg.kf, reg.nr, reg.betas, reg.vm
        r_global = []
        regi = np.zeros((reg.kr, 2))
        for vari in np.arange(kr):
            r_vari = buildR1var(vari, kr, kf, nr)
            r_global.append(r_vari)
            q = np.zeros((r_vari.shape[0], 1))
            regi[vari, :] = wald_test(betas, r_vari, q, vm)
        r_global = np.vstack(tuple(r_global))
        q = np.zeros((r_global.shape[0], 1))
        joint = wald_test(betas, r_global, q, vm)
        self.joint = joint
        self.regi = regi

class Wald:
    '''
    Chi sq. Wald statistic to test for restriction of coefficients.
    Implementation following Greene [1]_ eq. (17-24), p. 488
    ...

    Arguments
    =========
    reg     : regression object
              Regression object from PySAL.spreg
    r       : array
              Array of dimension Rxk (R being number of restrictions) with constrain setup.
    q       : array
              Rx1 array with constants in the constraint setup. See Greene
              [1]_ for reference.

    Attributes
    ==========
    w       : float
              Wald statistic
    pvalue  : float
              P value for Wald statistic calculated as a Chi sq. distribution
              with R degrees of freedom

    References
    ==========
    .. [1] W. Greene. 2003. Econometric Analysis (Fifth Edtion). Prentice Hall, Upper
       Saddle River. 
    '''
    def __init__(self, reg, r, q=None):
        if not q:
            q = np.zeros((r.shape[0], 1))
        self.w, self.pvalue = wald_test(reg.betas, r, q, reg.vm)

def wald_test(betas, r, q, vm):
    '''
    Chi sq. Wald statistic to test for restriction of coefficients.
    Implementation following Greene [1]_ eq. (17-24), p. 488
    ...

    Arguments
    =========
    betas   : array
              kx1 array with coefficient estimates
    r       : array
              Array of dimension Rxk (R being number of restrictions) with constrain setup.
    q       : array
              Rx1 array with constants in the constraint setup. See Greene
              [1]_ for reference.
    vm      : array
              kxk variance-covariance matrix of coefficient estimates

    Returns
    =======
    w       : float
              Wald statistic
    pvalue  : float
              P value for Wald statistic calculated as a Chi sq. distribution
              with R degrees of freedom

    References
    ==========
    .. [1] W. Greene. 2003. Econometric Analysis (Fifth Edtion). Prentice Hall, Upper
       Saddle River. 
    '''
    rbq = np.dot(r, betas) - q
    rvri = la.inv(np.dot(r, np.dot(vm, r.T)))
    w = np.dot(rbq.T, np.dot(rvri, rbq))[0][0]
    df = r.shape[0]
    pvalue = chisqprob(w, df)
    return w, pvalue

def buildR(kr, kf, nr):
    '''
    Build R matrix to globally test for spatial heterogeneity across regimes.
    The constraint setup reflects the null every beta is the same
    across regimes
    ...

    Arguments
    =========
    kr      : int
              Number of variables that vary across regimes ("regimized")
    kf      : int
              Number of variables that do not vary across regimes ("fixed" or
              global)
    nr      : int
              Number of regimes

    Returns
    =======
    R       : array
              Array with constrain setup to test stability across regimes of
              one variable
    '''
    return np.vstack(tuple(map(buildR1var, np.arange(kr), [kr]*kr, [kf]*kr, [nr]*kr)))

def buildR1var(vari, kr, kf, nr):
    '''
    Build R matrix to test for spatial heterogeneity across regimes in one
    variable. The constraint setup reflects the null betas for variable 'vari'
    are the same across regimes
    ...

    Arguments
    =========
    vari    : int
              Position of the variable to be tested (order in the sequence of
              variables per regime)
    kr      : int
              Number of variables that vary across regimes ("regimized")
    kf      : int
              Number of variables that do not vary across regimes ("fixed" or
              global)
    nr      : int
              Number of regimes

    Returns
    =======
    R       : array
              Array with constrain setup to test stability across regimes of
              one variable
    '''
    ncols = (kr * nr)
    nrows = nr - 1
    r = np.zeros((nrows, ncols), dtype=int)
    rbeg = 0
    cbeg = vari
    r[rbeg: rbeg+nrows , cbeg] = 1
    for j in np.arange(nrows):
        r[rbeg+j, kr + cbeg + j*kr] = -1
    return np.hstack( (r, np.zeros((nrows, kf), dtype=int)) )

def regimeX_setup(x, regimes, cols2regi, constant=False):
    '''
    Flexible full setup of a regime structure

    NOTE: constant term, if desired in the model, should be included in the x
    already
    ...

    Arguments
    =========
    x           : np.array
                  Dense array of dimension (n, k) with values for all observations
                  IMPORTANT: constant term (if desired in the model) should be
                  included
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
    '''
    n, k = x.shape
    cols2regi = np.array(cols2regi)
    if len(set(cols2regi))==1:
        xsp = x2xsp(x, regimes)
    else:
        not_regi = x[:, np.where(cols2regi==False)[0]]
        regi_subset = x[:, np.where(cols2regi)[0]]
        regi_subset = x2xsp(regi_subset, regimes)
        xsp = SP.hstack( (regi_subset, SP.csr_matrix(not_regi)) , format='csr')
    return xsp

def set_name_x_regimes(name_x, regimes, constant_regi, cols2regi):
    '''
    Generate the set of variable names in a regimes setup, according to the
    order of the betas

    NOTE: constant term, if desired in the model, should be included in the x
    already
    ...

    Arguments
    =========
    name_x          : list/None
                      If passed, list of strings with the names of the
                      variables aligned with the original dense array x
                      IMPORTANT: constant term (if desired in the model) should be
                      included
    regimes         : list
                      list of n values with the mapping of each observation to a
                      regime. Assumed to be aligned with 'x'.
    constant_regi   : [False, 'one', 'many']
                      Switcher controlling the constant term setup. It may take
                      the following values:
                    
                         *  False: no constant term is appended in any way
                         *  'one': a vector of ones is appended to x and held
                                   constant across regimes
                         * 'many': a vector of ones is appended to x and considered
                                   different per regime
    cols2regi       : list
                      List of k booleans indicating whether each column should be
                      considered as different per regime (True) or held constant
                      across regimes (False)
    Returns
    =======
    name_x_regi
    '''
    k = len(cols2regi)
    if constant_regi:
        k -= 1
    regimes_set = list(set(regimes))
    if not name_x:
        name_x = ['var_'+str(i+1) for i in range(k)]
    if constant_regi:
        name_x.insert(0, 'CONSTANT')
    nxa = np.array(name_x)
    c2ra = np.array(cols2regi)
    vars_regi = nxa[np.where(c2ra==True)]
    vars_glob = nxa[np.where(c2ra==False)]

    name_x_regi = []
    for r in regimes_set:
        rl = ['%s_-_%s'%(str(r), i) for i in vars_regi]
        name_x_regi.extend(rl)
    name_x_regi.extend(['Global_-_%s'%i for i in vars_glob])
    return name_x_regi

def regimes_printout(model):
    stds = diagnostics.se_betas(model)
    tp = np.array(diagnostics.t_stat(model))
    res = pd.DataFrame({'Coefficient': pd.Series(model.betas.flatten()), \
            'Std. Error': pd.Series(stds) , \
            'T-Stat': pd.Series(tp[:, 0].flatten()), \
            'P-value': pd.Series(tp[:, 1].flatten())})
    res = res.reindex(columns = ['Coefficient', 'Std. Error', 'T-Stat', 'P-value'])
    inds = []
    for lbl in model.name_x:
        r, v = lbl.split('_-_')
        inds.append((r, v))
    inds = np.array(inds)
    res['Regime'] = inds[:, 0]
    res['Variable'] = inds[:, 1]
    res = res.set_index(['Regime', 'Variable'])
    res[''] = res['P-value'].apply(star_significance)
    return res

def star_significance(p):
    if p < 0.001:
        return '***'
    elif p > 0.001 and p <= 0.005:
        return '** '
    elif p > 0.005 and p <= 0.1:
        return '*  '
    else:
        return ''

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
    np.random.seed(123)
    # Data Setup
    #n, kr, kf, r, constant = (1500000, 8, 1, 50, 'many')
    #n, kr, kf, r, constant = (50000, 16, 15, 359, 'many')
    #n, kr, kf, r = (50000, 11, 0, 359)
    n, kr, kf, r, constant = (1000, 3, 0, 2, 'many')
    print('Using setup with n=%i, kr=%i, kf=%i, %i regimes and %s constant(s)' \
            %(n, kr, kf, r, constant))
    k = kr + kf
    x = np.random.random((n, k))
    #x[:50, :] = (x[:50, :] + 100) ** 2
    x[:500, 2:] = np.random.normal(loc=10, size=(500, k-2)) + 10000
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
    #print('Full setup created in %.4f seconds'%(t1-t0))
    R = buildR(kr+1, kf, r)

    # Regression and Chow test
    ols = BaseOLS(y, xsp, constant=True)
    print ols.betas
    '''
    ols.kr = kr
    ols.kf = kf
    if constant == 'many':
        ols.kr += 1
    else:
        ols.kf += 1
    ols.nr = r
    t2 = time.time()
    print('OLS run in %.4f seconds'%(t2-t1))
    chow = Chow_sp(ols)

    # columbus test (against R::aod wald.test)
    import pandas as pd
    db = pd.read_csv('examples/columbus.csv')
    y = db['HOVAL,N,9,6'].values.reshape((len(db), 1))
    name_x = ['INC,N,9,6', 'DISCBD,N,8,6']
    x = db[name_x].as_matrix()
    lm = ps.spreg.OLS(y, x, name_x=name_x)
    R = np.array([[1, 0, 0]])
    wald = Wald(lm, R)
    '''

