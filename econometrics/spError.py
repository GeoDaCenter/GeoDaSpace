"""
Spatial Error Models module
"""
import pysal
import time
import scipy.optimize as op
from scipy import sparse as SP
from scipy.stats import norm
import numpy as np
import numpy.linalg as la
import gmm_utils as GMM
from gmm_utils import get_A1, get_spFilter
import ols as OLS
from diagnostics import se_betas


class GSLS:
    """
    Generalized Spatial Least Squares (OLS + GMM) as in Kelejian and Prucha
    (1998) [1]_ and Kelejian and Prucha (1999) [2]_
    ...

    Parameters
    ----------

    x           : array
                  nxk array of independent variables (assumed to be aligned with y)
    y           : array
                  nx1 array of dependent variable
    w           : W
                  Spatial weights instance assumed to be row-standardized

    Attributes
    ----------

    betas       : array
                  kx1 array with estimated coefficients
    se_betas    : array
                  kx1 array with standard errors for estimated coefficients
                  NOTE: it corrects by sqrt( (n-k)/n ) as in R's spdep
    z           : array
                  kx1 array with estimated coefficients divided by the standard errors
    pvals       : array
                  kx1 array with p-values of the estimated coefficients
    lamb        : float
                  Point estimate for the spatially autoregressive parameter in
                  the error
    u           : array
                  Vector of residuals
    sig2        : float
                  Sigma squared for the residuals of the transformed model (as
                  in R's spdep)

    References
    ----------

    .. [1] Kelejian, H.R., Prucha, I.R. (1998) "A generalized spatial
    two-stage least squares procedure for estimating a spatial autoregressive
    model with autoregressive disturbances". The Journal of Real State
    Finance and Economics, 17, 1.

    .. [2] Kelejian, H.R., Prucha, I.R. (1999) "A Generalized Moments
    Estimator for the Autoregressive Parameter in a Spatial Model".
    International Economic Review, 40, 2.

    Examples
    --------

    >>> dbf = pysal.open('examples/columbus.dbf','r')
    >>> y = np.array([dbf.by_col('HOVAL')]).T
    >>> x = np.array([dbf.by_col('INC'), dbf.by_col('CRIME')]).T
    >>> w = pysal.open('examples/columbus.GAL', 'r').read() 
    >>> w.transform='r'
    >>> model = GSLS(x, y, w)
    >>> np.around(model.betas, decimals=6)
    array([[ 47.694634],
           [  0.710453],
           [ -0.550527]])
    >>> np.around(model.se_betas, decimals=6)
    array([[ 12.412039],
           [  0.504443],
           [  0.178496]])
    >>> np.around(model.z, decimals=6)
    array([[ 3.842611],
           [ 1.408391],
           [-3.084247]])
    >>> np.around(model.pvals, decimals=6)
    array([[  1.22000000e-04],
           [  1.59015000e-01],
           [  2.04100000e-03]])
    >>> np.around(model.lamb, decimals=6)
    0.32573000000000002
    >>> np.around(model.sig2, decimals=6)
    198.559595

    """
    def __init__(self, x, y, w):
        w.A1 = get_A1(w.sparse)

        x = np.hstack((np.ones(y.shape),x))
        n, k = x.shape

        #1a. OLS --> \tilde{betas}
        ols = OLS.OLS_dev(x,y, constant=False)

        #1b. GMM --> \tilde{\lambda1}
        moments = GMM.momentsGSLS(w, ols.u)
        lambda1 = GMM.optimizer_gsls(moments)[0][0]

        #2a. OLS -->\hat{betas}
        xs,ys = get_spFilter(w, lambda1, x),get_spFilter(w, lambda1, y)

        ols = OLS.OLS_dev(xs,ys, constant=False)

        #Output
        self.betas = ols.betas
        self.lamb = lambda1
        self.sig2 = ols.sig2n
        self.u = ols.u
        self.se_betas, self.z, self.pvals = self._inference(ols)

    def _inference(self, ols):
        """
        Inference for estimated coefficients
        Coded as in GMerrorsar from R (which matches) using se_betas from diagnostics module
        """
        c = np.sqrt((ols.n-ols.k) / float(ols.n))
        ses = np.array([se_betas(ols) * c]).T
        zs = ols.betas / ses
        pvals = norm.sf(abs(zs)) * 2.
        return [ses, zs, pvals]
        
def _test():
    import doctest
    doctest.testmod()

if __name__ == '__main__':

    _test()

    dbf = pysal.open('examples/columbus.dbf','r')
    y = np.array([dbf.by_col('HOVAL')]).T
    x = np.array([dbf.by_col('INC'), dbf.by_col('CRIME')]).T
    w = pysal.open('examples/columbus.GAL', 'r').read()
    w.transform='r' #Needed to match R

    """
    print '\n\tGSLS model Example'
    model = GSLS(x, y, w)
    print '\n### Betas ###'
    print model.betas
    print '\n### Std. Errors ###'
    print model.se_betas
    print '\n### z-Values ###'
    print model.z
    print '\n### p-Values ###'
    print model.pvals
    print '\n### Lambda ###'
    print model.lamb
    print '\n### Sig2 ###'
    print model.sig2
    """

