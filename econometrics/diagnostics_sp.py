"""
Spatial diagnostics module

ToDo:

    * Checking against R's spdep differs in:
        * Moran's variance
    * Document Moran
"""

from scipy.stats.stats import chisqprob
from scipy.stats import norm
from ols import OLS_dev as OLS
#from twosls import TSLS_dev
#from twosls_sp import STSLS_dev
import numpy as np
import numpy.linalg as la
import pysal

class LMtests:
    """
    Lagrange Multiplier tests. Implemented as presented in Anselin et al.
    (1996) [1]_
    ...

    Attributes
    ----------

    x           : array
                  nxk array of independent variables (assumed to be aligned with y)
    y           : array
                  nx1 array of dependent variable
    w           : W
                  Spatial weights instance (requires 'S' and 'A1') assumed to
                  be row-standardized
    constant    : boolean
                  If true it appends a vector of ones to the independent variables
                  to estimate intercept (set to True by default)
    tests       : list
                  Lists of strings with the tests desired to be performed.
                  Values may be:
                  
                  * 'all': runs all the options (default)
                  * 'lme': LM error test
                  * 'rlme': Robust LM error test
                  * 'lml' : LM lag test
                  * 'rlml': Robust LM lag test

    Parameters
    ----------

    lme         : tuple
                  (Only if 'lme' or 'all' was in tests). Pair of statistic and
                  p-value for the LM error test.
    lml         : tuple
                  (Only if 'lml' or 'all' was in tests). Pair of statistic and
                  p-value for the LM lag test.
    rlme        : tuple
                  (Only if 'rlme' or 'all' was in tests). Pair of statistic
                  and p-value for the Robust LM error test.
    rlml        : tuple
                  (Only if 'rlml' or 'all' was in tests). Pair of statistic
                  and p-value for the Robust LM lag test.
    sarma       : tuple
                  (Only if 'rlml' or 'all' was in tests). Pair of statistic
                  and p-value for the SARMA test.

    References
    ----------
    .. [1] Anselin, L., Bera, A. K., Florax, R., Yoon, M. J. (1996) "Simple
       diagnostic tests for spatial dependence". Regional Science and Urban
       Economics, 26, 77-104.

    Examples
    --------

    >>> import numpy as np
    >>> import pysal
    >>> csv = pysal.open('examples/columbus.dbf','r')
    >>> y = np.array([csv.by_col('HOVAL')]).T
    >>> x = np.array([csv.by_col('INC'), csv.by_col('CRIME')]).T
    >>> w = pysal.open('examples/columbus.gal', 'r').read()
    >>> w.transform='r'
    >>> from ols import OLS_dev as OLS
    >>> wy = w.sparse * y
    >>> ols = OLS(x, y)
    >>> lms = LMtests(x, y, w)
    >>> lms.lme
    (3.992947267432295, 0.045691076106858394)
    >>> lms.lml
    (1.8194874340454195, 0.17737430099228549)
    >>> lms.rlme
    (2.8712886542421709, 0.090172642054033358)
    >>> lms.rlml
    (0.69782882085529507, 0.4035142049850427)
    >>> lms.sarma
    (4.6907760882875902, 0.095810016400331821)
    """
    def __init__(self, x, y, w, constant=True, tests=['all']):
        if w.transform != 'R':
            w.transform = 'r'
            print '\nYour W object has been row-standardized\n'
        ols = OLS(x, y, constant=constant)
        cache = spDcache(ols, w)
        if tests == ['all']:
            tests = ['lme', 'lml','rlme', 'rlml', 'sarma']
        if 'lme' in tests:
            self.lme = lmErr(ols, w, cache)
        if 'lml' in tests:
            self.lml = lmLag(ols, w, cache)
        if 'rlme' in tests:
            self.rlme = rlmErr(ols, w, cache)
        if 'rlml' in tests:
            self.rlml = rlmLag(ols, w, cache)
        if 'sarma' in tests:
            self.sarma = lmSarma(ols, w, cache)

class AKtest:
    """
    Moran's I test of spatial autocorrelation for IV estimation.
    Implemented following the original reference Anselin and Kelejian
    (1997) [1]_
    ...

    Parameters
    ----------

    w           : W
                  Spatial weights instance (requires 'S' and 'A1') assumed to
                  be row-standardized
    x           : array
                  nxk array of independent variables, including endogenous
                  variables (assumed to be aligned with y)
    y           : array
                  nx1 array of dependent variable
    h           : array
                  nxl array of instruments; typically this includes all
                  exogenous variables from x and instruments. If case='gen' and set
                  to None then only spatial lagged 
    constant    : boolean
                  If true it appends a vector of ones to the independent variables
                  to estimate intercept (set to True by default)
    case        : string
                  Flag for special cases (default to 'nosp'):
                    * 'nsp': Only NO spatial end. reg.
                    * 'gen': General case (spatial lag + end. reg.)
    w_lags      : int
                  [Only if case=0] Number of spatial lags of the exogenous
                  variables. Kelejian et al. (2004) [2]_ recommends w_lags=2, which
                  is the default.
    constant    : boolean
                  [Only if case=0] If true it appends a vector of ones to the
                  independent variables to estimate intercept (set to True by
                  default)

    robust      : string
                  [Only if case=0] If 'white' then a White consistent
                  estimator of the variance-covariance matrix is given. If 'gls' then
                  generalized least squares is performed resulting in new
                  coefficient estimates along with a new variance-covariance
                  matrix. 


    Attributes
    ----------

    mi          : float
                  Moran's I statistic for IV residuals
    ak          : float
                  Square of corrected Moran's I for residuals::
                    
                  .. math::

                        ak = \dfrac{N \times I^*}{\phi^2}

                  Note: if case='nsp' then it simplifies to the LMerror

    p           : float
                  P-value of the test

    References
    ----------

    .. [1] Anselin, L., Kelejian, H. (1997) "Testing for spatial error
    autocorrelation in the presence of endogenous regressors". Interregional
    Regional Science Review, 20, 1.

    .. [2] Kelejian, H.H., Prucha, I.R. and Yuzefovich, Y. (2004)
    "Instrumental variable estimation of a spatial autorgressive model with
    autoregressive disturbances: large and small sample results". Advances in
    Econometrics, 18, 163-198.
            """

    def __init__(self, w, x, y, h=None, case='nosp', w_lags=2, constant=True,
            robust=None):
        if case == 'gen':
            iv = STSLS_dev(x, y, w, h, w_lags=2, constant=constant, robust=robust)
            cache = spDcache(iv, w)
            self.mi, self.ak, self.p = akTest(iv, w, cache)
        elif case == 'nsp':
            iv = TSLS_dev(x, y, h, constant=constant)
            cache = spDcache(iv, w)
            self.mi = get_mI(iv, w, cache)
            self.ak, self.p = lmErr(iv, w, cache)
        else:
            print """\n
            Fix the optional argument 'case' to match the requirements:
                * 0: General case (spatial lag + end. reg.)
                * 1: No spatial end. reg.
            \n"""

class MoranRes:
    def __init__(self, x, y, w, constant=True):
        ols = OLS(x, y, constant=constant)
        cache = spDcache(ols, w)
        self.I = get_mI(ols, w, cache)
        self.eI = get_eI(ols, w, cache)
        self.vI = get_vI(ols, w, self.eI, cache)
        self.zI, self.p_norm = get_zI(self.I, self.eI, self.vI)

class spDcache:
    """
    Class to compute reusable pieces in the spatial diagnostics module
    ...

    Parameters
    ----------

    reg         : OLS_dev, TSLS_dev, STSLS_dev
                  Instance from a regression class
    w           : W
                  Spatial weights instance (requires 'S' and 'A1') assumed to
                  be row-standardized

    Attributes
    ----------

    j           : array
                  1x1 array with the result from:

                  .. math::

                        J = \dfrac{1}{[(WX\beta)' M (WX\beta) + T \sigma^2]}

    wu          : array
                  nx1 array with spatial lag of the residuals

    utwuDs      : array
                  1x1 array with the result from:

                  .. math::

                        utwuDs = \dfrac{u' W u}{\tilde{\sigma^2}}

    utwyDs      : array
                  1x1 array with the result from:

                  .. math::

                        utwyDs = \dfrac{u' W y}{\tilde{\sigma^2}}


    t           : array
                  1x1 array with the result from :

                  .. math::

                        T = tr[(W' + W) W]

    mw          : csr_matrix
                  scipy sparse matrix results of multiplying reg.M and W
    trMw        : float
                  Trace of mw

    """
    def __init__(self,reg, w):
        self.reg = reg
        self.w = w
        self._cache = {}
    @property
    def j(self):
        if 'j' not in self._cache:
            wxb = self.w.sparse * self.reg.predy
            num = np.dot(wxb.T, np.dot(self.reg.m, wxb)) + (self.t * self.reg.sig2)
            den = self.reg.n * self.reg.sig2
            self._cache['j'] = num / den
        return self._cache['j']
    @property
    def wu(self):
        if 'wu' not in self._cache:
            self._cache['wu'] = self.w.sparse * self.reg.u
        return self._cache['wu']
    @property
    def utwuDs(self):
        if 'utwuDs' not in self._cache:
            res = np.dot(self.reg.u.T, self.wu) / self.reg.sig2
            self._cache['utwuDs'] = res
        return self._cache['utwuDs']
    @property
    def utwyDs(self):
        if 'utwyDs' not in self._cache:
            res = np.dot(self.reg.u.T, self.w.sparse * self.reg.y)
            self._cache['utwyDs'] = res / self.reg.sig2
        return self._cache['utwyDs']
    @property
    def t(self):
        if 't' not in self._cache:
            prod = (self.w.sparse.T + self.w.sparse) * self.w.sparse 
            self._cache['t'] = np.sum(prod.diagonal())
        return self._cache['t']
    @property
    def mw(self):
        if 'mw' not in self._cache:
            self._cache['mw'] = self.reg.m * self.w.sparse
        return self._cache['mw']
    @property
    def trMw(self):
        if 'trMw' not in self._cache:
            self._cache['trMw'] = np.sum(self.mw.diagonal())
        return self._cache['trMw']
    @property
    def AB(self):
        """
        Computes A and B matrices as in Cliff-Ord 1981, p. 203
        """
        if 'AB' not in self._cache:
            U = (self.w.sparse + self.w.sparse.T) / 2.
            z = U * self.reg.x
            c1 = np.dot(self.reg.x.T, z)
            c2 = np.dot(z.T, z)
            G = self.reg.xtxi
            A = np.dot(G, c1)
            B = np.dot(G, c2)
            self._cache['AB'] = [A, B]
        return self._cache['AB']
    @property
    def trA(self):
        if 'trA' not in self._cache:
            self._cache['trA'] = np.sum(self.AB[0].diagonal())
        return self._cache['trA']


def lmErr(reg, w, spDcache):
    """
    LM error test. Implemented as presented in eq. (9) of Anselin et al.
    (1996) [1]_
    ...

    Attributes
    ----------

    reg         : OLS_dev, TSLS_dev, STSLS_dev
                  Instance from a regression class
    w           : W
                  Spatial weights instance (requires 'S' and 'A1') assumed to
                  be row-standardized
    spDcache     : spDcache
                  Instance of spDcache class

    Returns
    -------

    lme         : tuple
                  Pair of statistic and p-value for the LM error test.

    References
    ----------
    .. [1] Anselin, L., Bera, A. K., Florax, R., Yoon, M. J. (1996) "Simple
       diagnostic tests for spatial dependence". Regional Science and Urban
       Economics, 26, 77-104.
    """
    lm = spDcache.utwuDs**2 / spDcache.t
    pval = chisqprob(lm, 1)
    return (lm[0][0], pval[0][0])

def lmLag(ols, w, spDcache):
    """
    LM lag test. Implemented as presented in eq. (13) of Anselin et al.
    (1996) [1]_
    ...

    Attributes
    ----------

    ols         : OLS_dev
                  Instance from an OLS_dev regression 
    w           : W
                  Spatial weights instance (requires 'S' and 'A1') assumed to
                  be row-standardized
    spDcache     : spDcache
                  Instance of spDcache class

    Returns
    -------

    lml         : tuple
                  Pair of statistic and p-value for the LM lag test.

    References
    ----------
    .. [1] Anselin, L., Bera, A. K., Florax, R., Yoon, M. J. (1996) "Simple
       diagnostic tests for spatial dependence". Regional Science and Urban
       Economics, 26, 77-104.
    """
    lm = spDcache.utwyDs**2 / (ols.n * spDcache.j)
    pval = chisqprob(lm, 1)
    return (lm[0][0], pval[0][0])

def rlmErr(ols, w, spDcache):
    """
    Robust LM error test. Implemented as presented in eq. (8) of Anselin et al. (1996) [1]_ 

    NOTE: eq. (8) has an errata, the power -1 in the denominator should be inside the square bracket.
    ...

    Attributes
    ----------

    ols         : OLS_dev
                  Instance from an OLS_dev regression 
    w           : W
                  Spatial weights instance (requires 'S' and 'A1') assumed to
                  be row-standardized
    spDcache     : spDcache
                  Instance of spDcache class

    Returns
    -------

    rlme        : tuple
                  Pair of statistic and p-value for the Robust LM error test.

    References
    ----------
    .. [1] Anselin, L., Bera, A. K., Florax, R., Yoon, M. J. (1996) "Simple
       diagnostic tests for spatial dependence". Regional Science and Urban
       Economics, 26, 77-104.
    """
    nj = ols.n * spDcache.j
    num = (spDcache.utwuDs - (spDcache.t * spDcache.utwyDs) / nj)**2
    den = spDcache.t * (1. - (spDcache.t / nj))
    lm = num / den
    pval = chisqprob(lm, 1)
    return (lm[0][0], pval[0][0])

def rlmLag(ols, w, spDcache):
    """
    Robust LM lag test. Implemented as presented in eq. (12) of Anselin et al.
    (1996) [1]_
    ...

    Attributes
    ----------

    ols             : OLS_dev
                      Instance from an OLS_dev regression 
    w               : W
                      Spatial weights instance (requires 'S' and 'A1') assumed to
                      be row-standardized
    spDcache        : spDcache
                      Instance of spDcache class

    Returns
    -------

    rlml            : tuple
                      Pair of statistic and p-value for the Robust LM lag test.

    References
    ----------
    .. [1] Anselin, L., Bera, A. K., Florax, R., Yoon, M. J. (1996) "Simple
       diagnostic tests for spatial dependence". Regional Science and Urban
       Economics, 26, 77-104.
    """
    lm = (spDcache.utwyDs - spDcache.utwuDs)**2 / ((ols.n * spDcache.j) - spDcache.t)
    pval = chisqprob(lm, 1)
    return (lm[0][0], pval[0][0])

def lmSarma(ols, w, spDcache):
    """
    LM error test. Implemented as presented in eq. (15) of Anselin et al.
    (1996) [1]_
    ...

    Attributes
    ----------

    ols         : OLS_dev
                  Instance from an OLS_dev regression 
    w           : W
                  Spatial weights instance (requires 'S' and 'A1') assumed to
                  be row-standardized
    spDcache     : spDcache
                  Instance of spDcache class

    Returns
    -------

    sarma       : tuple
                  Pair of statistic and p-value for the LM sarma test.

    References
    ----------
    .. [1] Anselin, L., Bera, A. K., Florax, R., Yoon, M. J. (1996) "Simple
       diagnostic tests for spatial dependence". Regional Science and Urban
       Economics, 26, 77-104.
    """

    first = (spDcache.utwyDs - spDcache.utwuDs)**2 / (w.n * spDcache.j - spDcache.t)
    secnd = spDcache.utwuDs**2 / spDcache.t
    lm = first + secnd
    pval = chisqprob(lm, 2)
    return (lm[0][0], pval[0][0])

def akTest(iv, w, spDcache):
    """
    Computes AK-test for the general case (end. reg. + sp. lag)
    ...

    Parameters
    ----------

    iv          : STSLS_dev
                  Instance from spatial 2SLS regression
    w           : W
                  Spatial weights instance (requires 'S' and 'A1') assumed to
                  be row-standardized
   spDcache     : spDcache
                  Instance of spDcache class

    Attributes
    ----------
    mi          : float
                  Moran's I statistic for IV residuals
    ak          : float
                  Square of corrected Moran's I for residuals::
                    
                  .. math::

                        ak = \dfrac{N \times I^*}{\phi^2}

    p           : float
                  P-value of the test

    ToDo:
        * Code in as Nancy
        * Compare both
    """
    mi = get_mI(iv, w, spDcache)
    # Phi2
    etwz = np.dot(iv.u.T, (w.sparse * iv.z))
    p = np.dot(iv.x, np.dot(iv.xtxi, iv.x.T))
    ztpz = np.dot(iv.z.T, np.dot(p, iv.z))
    nztpzi = w.n * la.inv(ztpz)
    a = np.dot((etwz / w.n), np.dot(nztpzi, (etwz.T / w.n)))
    s12 = (w.s0 / w.n)**2
    s2 = (2. * w.trcW2 + w.trcWtW) / w.n
    phi2 = (s2 / 2. * s12) + (4. / (s12 * iv.sig2)) * a
    ak = w.n * mi**2 / phi2 # ak = (N^{1/2} * I* / phi)^2
    pval = chisqprob(ak, 1)
    return (mi, ak[0][0], pval[0][0])

def akTest_legacy(iv, w, spDcache):
    """
    Computes AK-test for the general case (end. reg. + sp. lag) coded as in
    GeoDaSpace legacy (Nancy's code)
    ...

    Parameters
    ----------

    iv          : STSLS_dev
                  Instance from spatial 2SLS regression
    w           : W
                  Spatial weights instance (requires 'S' and 'A1') assumed to
                  be row-standardized
   spDcache     : spDcache
                  Instance of spDcache class

    Attributes
    ----------
    ak          : tuple
                  Pair of statistic and p-value for the AK test


    """
    ewe = np.dot(iv.u.T, spDcache.wu)
    mi1 = w.n * ewe / iv.utu
    mi = mi1 / w.s0
    t = w.trcWtW_WW
    wz = w.sparse * iv.z
    ZWe = np.dot(wz.T, iv.utu)

    hph = np.dot(iv.h.T, iv.h)
    ihph = la.inv(hph)
    zph = np.dot(iv.z.T, iv.h)
    z1 = np.dot(zph, ihph)
    hpz = np.transpose(zph)
    zhpzh = np.dot(z1,np.transpose(zph))
    izhpzh = la.inv(zhpzh)

    dZWe = np.dot(izhpzh, ZWe)
    eWZdZWe = np.dot(np.transpose(ZWe),dZWe)
    denom = 4.0 * w.n * eWZdZWe[0][0] / iv.utu
    mchi = mi1**2 / (t + denom)
    pmchi = chisqprob(mchi,1)
    return (mchi[0][0], pmchi[0][0])

def get_mI(reg, w, spDcache):
    """
    Moran's I statistic of spatial autocorrelation as showed in Cliff & Ord
    (1981) [1], p. 201-203
    ...

    Attributes
    ----------

    reg             : OLS_dev, TSLS_dev, STSLS_dev
                      Instance from a regression class
    w               : W
                      Spatial weights instance (requires 'S' and 'A1') assumed to
                      be row-standardized
    spDcache        : spDcache
                      Instance of spDcache class

    Returns
    -------

    moran           : float
                      Statistic Moran's I test.

    References
    ----------
    .. [1] Cliff, AD., Ord, JK. (1981) "Spatial processes: models & applications".
       Pion London
    """
    mi = (w.n * np.dot(reg.u.T, spDcache.wu)) / (w.s0 * reg.utu)
    return mi[0][0]

def get_eI(ols, w, spDcache):
    """
    Moran's I expectation as in Cliff & Ord 1981 (p. 201-203)
    """
    return -(w.n * spDcache.trA) / (w.s0 * (w.n - ols.k))

def get_vI(ols, w, ei, spDcache):
    """
    Moran's I variance coded as in Cliff & Ord 1981 (p. 201-203) and R's spdep
    """
    A = spDcache.AB[0]
    trA2 = np.dot(A, A)
    trA2 = np.sum(trA2.diagonal())

    B = spDcache.AB[1]
    trB = np.sum(B.diagonal()) * 4
    vi = (w.n**2 / (w.s0**2 * (w.n - ols.k) * (w.n - ols.k + 2))) * \
            (w.s1 + 2 * trA2 - trB - ((2 * (spDcache.trA**2)) / (w.n - ols.k)))
    return vi

def get_eI_m(ols, w, spDcache):
    """
    Moran's I expectation using matrix M
    """
    return (w.n * spDcache.trMw) / (w.s0 * (w.n - ols.k))

def get_vI_m(ols, w, ei, spDcache):
    """
    Moran's I variance using matrix M
    """
    trMwmwt = np.dot(spDcache.mw, (ols.m * w.sparse.T))
    trMwmwt = np.sum(trMwmwt.diagonal())
    #mw2 = spDcache.mw**2
    mw2 = np.dot(spDcache.mw, spDcache.mw)
    num = w.n**2 * (trMwmwt + np.sum(mw2.diagonal()) + spDcache.trMw**2)
    den = w.s0**2 * (((w.n - ols.k) * (w.n - ols.k + 2)) )
    return (num / den) - ei**2

def get_zI(I, ei, vi):
    """
    Standardized I

    Returns two-sided p-values as provided in the GeoDa family
    """
    z = abs((I - ei) / np.sqrt(vi))
    pval = norm.sf(z) * 2.
    return (z, pval)

def _test():
    import doctest
    doctest.testmod()

if __name__ == '__main__':
    _test()

    import numpy as np
    import pysal
    csv = pysal.open('examples/columbus.dbf','r')
    y = np.array([csv.by_col('HOVAL')]).T
    x = np.array([csv.by_col('INC'), csv.by_col('CRIME')]).T
    w = pysal.open('examples/columbus.gal', 'r').read()
    w.transform='r'
    ols = OLS(x, y)
    """
    iv = STSLS_dev(x, y, w, h=None, w_lags=2, constant=True, robust=None)
    cache = spDcache(iv, w)
    ak = akTest(iv, w, cache)[1]
    AK = AKtest(w, x, y, h=iv.h, case='nsp')
    akl = akTest_legacy(iv, w, cache)[0]
    print 'AK: %f\tAK_legacy: %f'%(ak, akl)
    """

