"""
Spatial diagnostics module
"""
from scipy.stats.stats import chisqprob
from ols import OLS_dev as OLS
import numpy as np

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

    References
    ----------
    .. [1] Anselin, L., Bera, A. K., Florax, R., Yoon, M. J. (1996) "Simple
       diagnostic tests for spatial dependence". Regional Science and Urban
       Economics, 26, 77-104.
    """
    def __init__(self, x, y, w, constant=True, tests=['all']):
        ols = OLS(x, y, constant=constant)
        cache = spDcache(ols, w)
        if tests == ['all']:
            tests = ['lme', 'lml','rlme', 'rlml']
        if 'lme' in tests:
            self.lme = lmErr(ols, w, cache)
        if 'lml' in tests:
            self.lml = lmLag(ols, w, cache)
        if 'rlme' in tests:
            self.rlme = rlmErr(ols, w, cache)
        if 'rlml' in tests:
            self.rlml = rlmLag(ols, w, cache)


def lmErr(ols, w, spDcache):
    """
    LM error test. Implemented as presented in eq. (9) of Anselin et al.
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
    Robust LM error test. Implemented as presented in eq. (8) of Anselin et al.
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
    den = spDcache.t / (1. - (spDcache.t * nj))
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

    ols         : OLS_dev
                  Instance from an OLS_dev regression 
    w           : W
                  Spatial weights instance (requires 'S' and 'A1') assumed to
                  be row-standardized
    spDcache     : spDcache
                  Instance of spDcache class

    Returns
    -------

    rlml        : tuple
                  Pair of statistic and p-value for the Robust LM lag test.

    References
    ----------
    .. [1] Anselin, L., Bera, A. K., Florax, R., Yoon, M. J. (1996) "Simple
       diagnostic tests for spatial dependence". Regional Science and Urban
       Economics, 26, 77-104.
    """
    lm = (spDcache.utwyDs - spDcache.utwuDs) / ((ols.n * spDcache.j) - spDcache.t)
    pval = chisqprob(lm, 1)
    return (lm[0][0], pval[0][0])

class spDcache:
    """
    Class to compute reusable pieces in LM tests
    ...

    Attributes
    ----------

    ols         : OLS_dev
                  Instance from an OLS_dev regression 
    w           : W
                  Spatial weights instance (requires 'S' and 'A1') assumed to
                  be row-standardized

    Parameters
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
                  scipy sparse matrix results of multiplying ols.M and W
    trMw        : float
                  Trace of mw

    """
    def __init__(self,ols, w):
        self.ols = ols
        self.w = w
        self._cache = {}
    @property
    def j(self):
        if 'j' not in self._cache:
            wxb = self.w.S * self.ols.predy
            self._cache['j'] = (np.dot(wxb.T, np.dot(self.ols.m, wxb)) + (self.t * self.ols.sig2)) / (self.ols.n * self.ols.sig2)
        return self._cache['j']
    @property
    def wu(self):
        if 'wu' not in self._cache:
            self._cache['wu'] = self.w.S * self.ols.u
        return self._cache['wu']
    @property
    def utwuDs(self):
        if 'utwuDs' not in self._cache:
            res = np.dot(self.ols.u.T, self.wu) / self.ols.sig2
            self._cache['utwuDs'] = res
        return self._cache['utwuDs']
    @property
    def utwyDs(self):
        if 'utwyDs' not in self._cache:
            res = np.dot(self.ols.u.T, self.w.S * self.ols.y) / self.ols.sig2
            self._cache['utwyDs'] = res
        return self._cache['utwyDs']
    @property
    def t(self):
        if 't' not in self._cache:
            prod = (self.w.S.T + self.w.S) * self.w.S 
            self._cache['t'] = np.sum(prod.diagonal())
        return self._cache['t']
    @property
    def mw(self):
        if 'mw' not in self._cache:
            self._cache['mw'] = ols.m * w.S
        return self._cache['mw']
    @property
    def trMw(self):
        if 'trMw' not in self._cache:
            self._cache['trMw'] = np.sum(self.mw.diagonal())
        return self._cache['trMw']

class MoranTest:
    def __init__(self, x, y, w, constant=True):
        ols = OLS(x, y, constant=constant)
        cache = spDcache(ols, w)
        self.I = get_mI(ols, w, cache)
        self.eI = get_eI(ols, w)
        self.vI = get_vI(ols, w, self.eI, cache)
        self.zI = get_zI(self.I, self.eI, self.vI)


def get_mI(ols, w, spDcache):
    return (w.n * np.dot(ols.u.T, spDcache.wu)) / (w.s0 * spDcache.utu)

def get_eI(ols, w):
    return (w.n * self.trMw) / (w.s0 * (w.n - ols.k))

def get_vI(ols, w, ei, spDcache):
    trMwmwt = spDcache.mw * (ols.m * w.S.T)
    trMwmwt = np.sum(trMwmwt.diagonal())
    mw2 = spDcache.mw**2
    num = n**2 * (trMwmwt + np.sum(mw2.diagonal()) + spDcache.trMw**2)
    den = w.s0**2 * (((w.n - ols.k)(w.n - ols.k + 2)) - ei**2)
    return num / den

def get_zI(I, ei, vi):
    return (I - ei) / np.sqrt(vi)

