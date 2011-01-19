"""
Spatial diagnostics module
"""

from scipy.stats.stats import chisqprob
from scipy.stats import norm
import numpy as np
import numpy.linalg as la
import pysal

__all__ = ['LMtests', 'MoranRes', 'spDcache'] 

class LMtests:
    """
    Lagrange Multiplier tests. Implemented as presented in Anselin et al.
    (1996) [1]_
    ...

    Attributes
    ----------

    ols         : OLS
                  OLS regression object
    w           : W
                  Spatial weights instance
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
    >>> from ols import OLS
    >>> csv = pysal.open('examples/columbus.dbf','r')
    >>> y = np.array([csv.by_col('HOVAL')]).T
    >>> x = np.array([csv.by_col('INC'), csv.by_col('CRIME')]).T
    >>> w = pysal.open('examples/columbus.gal', 'r').read()
    >>> w.transform='r'
    >>> ols = OLS(y, x)
    >>> lms = LMtests(ols, w)
    >>> np.around(lms.lme, decimals=6)
    array([ 3.097094,  0.078432])
    >>> np.around(lms.lml, decimals=6)
    array([ 0.981552,  0.321816])
    >>> np.around(lms.rlme, decimals=6)
    array([ 3.209187,  0.073226])
    >>> np.around(lms.rlml, decimals=6)
    array([ 1.093645,  0.295665])
    >>> np.around(lms.sarma, decimals=6)
    array([ 4.190739,  0.123025])
    """
    def __init__(self, ols, w, tests=['all']):
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

class MoranRes:
    """
    Moran's I for spatial autocorrelation in residuals from OLS regression
    ...

    Parameters
    ----------

    ols         : OLS
                  OLS regression object
    w           : W
                  Spatial weights instance
    z           : boolean
                  If set to True computes attributes eI, vI and zI. Due to computational burden of vI, defaults to False.

    Attributes
    ----------
    I           : float
                  Moran's I statistic
    eI          : float
                  Moran's I expectation
    vI          : float
                  Moran's I variance
    zI          : float
                  Moran's I standardized value

    Examples
    --------

    >>> import numpy as np
    >>> import pysal
    >>> from ols import OLS
    >>> csv = pysal.open('examples/columbus.dbf','r')
    >>> y = np.array([csv.by_col('HOVAL')]).T
    >>> x = np.array([csv.by_col('INC'), csv.by_col('CRIME')]).T
    >>> w = pysal.open('examples/columbus.gal', 'r').read()
    >>> w.transform='r'
    >>> ols = OLS(y, x)
    >>> m = MoranRes(ols, w, z=True)
    >>> np.around(m.I, decimals=6)
    0.17130999999999999
    >>> np.around(m.eI, decimals=6)
    -0.034522999999999998
    >>> np.around(m.vI, decimals=6)
    0.0081300000000000001
    >>> np.around(m.zI, decimals=6)
    2.2827389999999999
    """
    def __init__(self, ols, w, z=False):
        cache = spDcache(ols, w)
        self.I = get_mI(ols, w, cache)
        if z:
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
                  Spatial weights instance

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

    trA         : float
                  Trace of A as in Cliff & Ord (1981)

    """
    def __init__(self,reg, w):
        self.reg = reg
        self.w = w
        self._cache = {}
    @property
    def j(self):
        if 'j' not in self._cache:
            wxb = self.w.sparse * self.reg.predy
            wxb2 = np.dot(wxb.T, wxb)
            xwxb = np.dot(self.reg.x.T, wxb)
            num1 = wxb2 - np.dot(xwxb.T, np.dot(self.reg.xtxi, xwxb))
            num = num1 + (self.t * self.reg.sig2n)
            den = self.reg.n * self.reg.sig2n
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
            res = np.dot(self.reg.u.T, self.wu) / self.reg.sig2n
            self._cache['utwuDs'] = res
        return self._cache['utwuDs']
    @property
    def utwyDs(self):
        if 'utwyDs' not in self._cache:
            res = np.dot(self.reg.u.T, self.w.sparse * self.reg.y)
            self._cache['utwyDs'] = res / self.reg.sig2n
        return self._cache['utwyDs']
    @property
    def t(self):
        if 't' not in self._cache:
            prod = (self.w.sparse.T + self.w.sparse) * self.w.sparse 
            self._cache['t'] = np.sum(prod.diagonal())
        return self._cache['t']
    @property
    def trA(self):
        if 'trA' not in self._cache:
            xtwx = np.dot(self.reg.x.T, pysal.lag_spatial(self.w, self.reg.x))
            mw = np.dot(self.reg.xtxi, xtwx)
            self._cache['trA'] = np.sum(mw.diagonal())
        return self._cache['trA']
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
                  Spatial weights instance
    spDcache    : spDcache
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
                  Spatial weights instance 
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
                  Spatial weights instance
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
                      Spatial weights instance 
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
                  Spatial weights instance
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
                      Spatial weights instance
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

def get_eI(ols, w, spDcache):
    """
    Moran's I expectation using matrix M
    """
    return - (w.n * spDcache.trA) / (w.s0 * (w.n - ols.k))

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

