"""
Spatial diagnostics module

ToDo:

    * Checking against R's spdep differs in:
        * Moran's variance
    * Focus on:
        * np.dot against * for sparse (Moran)
    * Document Moran
"""
from scipy.stats.stats import chisqprob
from ols import OLS_dev as OLS
import numpy as np
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
    >>> from spHetErr import get_S
    >>> csv = pysal.open('examples/columbus.dbf','r')
    >>> y = np.array([csv.by_col('HOVAL')]).T
    >>> x = np.array([csv.by_col('INC'), csv.by_col('CRIME')]).T
    >>> w = pysal.open('examples/columbus.gal', 'r').read()
    >>> w.transform='r'
    >>> w.S = get_S(w)
    >>> from ols import OLS_dev as OLS
    >>> wy = w.S * y
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

    Attributes
    ----------
    x           : array
                  nxk array of independent variables, including endogenous
                  variables (assumed to be aligned with y)
    y           : array
                  nx1 array of dependent variable
    h           : array
                  nxl array of instruments; typically this includes all
                  exogenous variables from x and instruments
    constant    : boolean
                  If true it appends a vector of ones to the independent variables
                  to estimate intercept (set to True by default)
    w           : W
                  Spatial weights instance (requires 'S' and 'A1') assumed to
                  be row-standardized
    case        : int
                  Flag for special cases (default to 0):
                    * 0: General case
                    * 1: No endogenous regressors
                    * 2: No spatial lag

    References
    ----------
    .. [1] Anselin, L., Kelejian, H. (1997) "Testing for spatial error
    autocorrelation in the presence of endogenous regressors". Interregional
    Regional Science Review, 20, 1.
            """

    def __init__(self, iv, w, case=0):
        if case == 0:
            pass
        elif case == 1:
            pass
        elif case ==2:
            pass
        else:
            print """\n
            Fix the optional argument 'case' to match the requirements:
                * 0: General case
                * 1: No endogenous regressors
                * 2: No spatial lag
            \n"""

class MoranRes:
    def __init__(self, x, y, w, constant=True):
        ols = OLS(x, y, constant=constant)
        cache = spDcache(ols, w)
        self.I = get_mI(ols, w, cache)
        self.eI = get_eI(ols, w, cache)
        self.vI = get_vI(ols, w, self.eI, cache)
        self.zI = get_zI(self.I, self.eI, self.vI)

class spDcache:
    """
    Class to compute reusable pieces in the spatial diagnostics module
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
            num = np.dot(wxb.T, np.dot(self.ols.m, wxb)) + (self.t * self.ols.sig2)
            den = self.ols.n * self.ols.sig2
            self._cache['j'] = num / den
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
            res = np.dot(self.ols.u.T, self.w.S * self.ols.y)
            self._cache['utwyDs'] = res / self.ols.sig2
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
            self._cache['mw'] = self.ols.m * self.w.S
        return self._cache['mw']
    @property
    def trMw(self):
        if 'trMw' not in self._cache:
            self._cache['trMw'] = np.sum(self.mw.diagonal())
        return self._cache['trMw']

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


def get_mI(ols, w, spDcache):
    """
    Moran's I statistic of spatial autocorrelation
    """
    return (w.n * np.dot(ols.u.T, spDcache.wu)) / (w.s0 * ols.utu)

def get_eI(ols, w, spDcache):
    """
    Moran's I expectation
    """
    return (w.n * spDcache.trMw) / (w.s0 * (w.n - ols.k))

def get_vI(ols, w, ei, spDcache):
    """
    Moran's I variance
    """
    trMwmwt = np.dot(spDcache.mw, (ols.m * w.S.T))
    trMwmwt = np.sum(trMwmwt.diagonal())
    #mw2 = spDcache.mw**2
    mw2 = np.dot(spDcache.mw, spDcache.mw)
    num = w.n**2 * (trMwmwt + np.sum(mw2.diagonal()) + spDcache.trMw**2)
    den = w.s0**2 * (((w.n - ols.k) * (w.n - ols.k + 2)) )
    return (num / den) - ei**2

def get_vIr(ols, w, ei, spDcache):
    """
    Moran's I variance coded as in R's spded
    """
    z = w.S * ols.x
    c1 = np.dot(ols.x.T, z)
    c2 = np.dot(z.T, z)
    c3 = np.dot(ols.xtxi, c1)
    trA = np.sum(c3.diagonal())
    trA2 = np.dot(c3, c3)
    trA2 = np.sum(trA2.diagonal())
    trB = 4 * np.dot(ols.xtxi, c2)
    trB = np.sum(trB.diagonal())
    vi = ((w.n**2 / (w.s0**2 * (w.n - ols.k) * (w.n - ols.k + 2))) * \
            (w.s1 + 2 * trA2 - trB - ((2 * (trA**2)) / (w.n - ols.k))))
    return vi

def get_zI(I, ei, vi):
    """
    Standardized I
    """
    return (I - ei) / np.sqrt(vi)

def _test():
    import doctest
    doctest.testmod()

if __name__ == '__main__':
    _test()

