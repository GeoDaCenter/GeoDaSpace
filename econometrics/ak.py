from scipy.stats.stats import chisqprob
from scipy.stats import norm
import numpy as np
import numpy.linalg as la
import pysal


from diagnostics_sp import spDcache, get_mI

class AKtest:
    """
    Moran's I test of spatial autocorrelation for IV estimation.
    Implemented following the original reference Anselin and Kelejian
    (1997) [1]_
    ...

    Parameters
    ----------

    w           : W
                  Spatial weights instance assumed to
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

def akTest(iv, w, spDcache):
    """
    Computes AK-test for the general case (end. reg. + sp. lag)
    ...

    Parameters
    ----------

    iv          : STSLS_dev
                  Instance from spatial 2SLS regression
    w           : W
                  Spatial weights instance assumed to
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
    p = la.inv(np.dot(iv.h.T, iv.h))
    p = np.dot(p, iv.h.T)
    p = np.dot(iv.h, p)
    ztpz = np.dot(iv.z.T, np.dot(p, iv.z))
    nztpzi = w.n * la.inv(ztpz)
    a = np.dot((etwz / w.n), np.dot(nztpzi, (etwz.T / w.n)))
    s12 = (w.s0 / w.n)**2
    # s2
    s2 = w.sparse + w.sparse.T
    s2 = s2 * s2
    s2 = np.sum(s2.diagonal()) / w.n

    phi2 = (s2 / 2. * s12) + (4. / (s12 * iv.sig2n)) * a
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
                  Spatial weights instance assumed to
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

