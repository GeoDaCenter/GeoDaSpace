from scipy.stats.stats import chisqprob
from scipy.stats import norm
import numpy as np
import numpy.linalg as la
import pysal


from pysal.spreg.diagnostics_sp import spDcache, get_mI

class AKtest:
    """
    Moran's I test of spatial autocorrelation for IV estimation.
    Implemented following the original reference Anselin and Kelejian
    (1997) [1]_
    ...

    Parameters
    ----------

    iv          : TSLS
                  Regression object from TSLS class
    w           : W
                  Spatial weights instance 
    case        : string
                  Flag for special cases (default to 'nosp'):
                    * 'nosp': Only NO spatial end. reg.
                    * 'gen': General case (spatial lag + end. reg.)

    Attributes
    ----------

    mi          : float
                  Moran's I statistic for IV residuals
    ak          : float
                  Square of corrected Moran's I for residuals::
                    
                  .. math::

                        ak = \dfrac{N \times I^*}{\phi^2}

                  Note: if case='nosp' then it simplifies to the LMerror
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

    def __init__(self, iv, w, case='nosp'):
        if case == 'gen':
            cache = spDcache(iv, w)
            self.mi, self.ak, self.p = akTest(iv, w, cache)
        elif case == 'nosp':
            cache = spDcache(iv, w)
            self.mi = get_mI(iv, w, cache)
            self.ak, self.p = lmErr(iv, w, cache)
        else:
            print """\n
            Fix the optional argument 'case' to match the requirements:
                * 'gen': General case (spatial lag + end. reg.)
                * 'nosp': No spatial end. reg.
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
                  Spatial weights instance 
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
    print '$$$$$'
    print 'nztpzi  ',nztpzi
    
    nztpzi = iv.varb
    print 'iv.varb ', iv.varb
    a = np.dot((etwz / w.n), np.dot(nztpzi, (etwz.T / w.n)))
    print '$$$$$$$$$'
    print 'a first time ',a
    a = np.dot(etwz,np.dot(nztpzi,etwz.T))
    a2 = a / w.n
    print 'a second time ',a2
    print 'equal? ', a == a2
    s12 = (w.s0 / w.n)**2
    print '$$$$$$$$$$$$$$$$$$$$$$$$'
    print 's12: ', s12
    print '$$$$$$$$$$$$$$$$$$$$$$$$'

    ## s2
    #s2 = w.sparse + w.sparse.T
    #s2 = s2 * s2
    #print np.sum(s2.diagonal()) 
    #print spDcache.t
    #s2 = np.sum(s2.diagonal()) / w.n
    s2 = spDcache.t / w.n
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
                  Spatial weights instance
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




def _test():
    import doctest
    doctest.testmod()

                     
if __name__ == '__main__':
    _test()    

