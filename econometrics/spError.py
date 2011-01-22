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
import pysal.spreg.ols as OLS
from pysal.spreg.diagnostics import se_betas


class GMSWLS:
    """
    Generalized Moments Spatially Weighted Least Squares (OLS + GMM) as in Kelejian and Prucha
    (1998) [1]_ and Kelejian and Prucha (1999) [2]_
    ...

    Parameters
    ----------

    x           : array
                  nxk array of independent variables (assumed to be aligned with y)
    y           : array
                  nx1 array of dependent variable
    w           : W
                  Spatial weights instance 

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
    >>> w = pysal.open('examples/columbus.gal', 'r').read() 
    >>> w.transform='r'
    >>> model = GMSWLS(x, y, w)
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
        ols = OLS.BaseOLS(y, x, constant=False)

        #1b. GMM --> \tilde{\lambda1}
        moments = self._momentsGMSWLS(w, ols.u)
        lambda1 = self._optimizer_gmswls(moments)[0][0]

        #2a. OLS -->\hat{betas}
        xs,ys = get_spFilter(w, lambda1, x),get_spFilter(w, lambda1, y)

        ols = OLS.BaseOLS(ys, xs, constant=False)

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

    def _momentsGMSWLS(self, w, u):

        u2 = np.dot(u.T, u)
        wu = w.sparse * u
        uwu = np.dot(u.T, wu)
        wu2 = np.dot(wu.T, wu)
        wwu = w.sparse * wu
        uwwu = np.dot(u.T, wwu)
        wwu2 = np.dot(wwu.T, wwu)
        wuwwu = np.dot(wu.T, wwu)
        wtw = w.sparse.T * w.sparse
        trWtW = np.sum(wtw.diagonal())

        g = np.array([[u2[0][0], wu2[0][0], uwu[0][0]]]).T / w.n

        G = np.array([[2 * uwu[0][0], -wu2[0][0], w.n], [2 * wuwwu[0][0], -wwu2[0][0], trWtW], [uwwu[0][0] + wu2[0][0], -wuwwu[0][0], 0.]]) / w.n

        return [G, g]

    def _optimizer_gmswls(self, moments):
        """
        Optimization of moments
        ...

        Parameters
        ----------

        moments     : _momentsGMSWLS
                      Instance of _momentsGMSWLS with G and g
        vcX         : array
                      Optional. 2x2 array with the Variance-Covariance matrix to be used as
                      weights in the optimization (applies Cholesky
                      decomposition). Set empty by default.

        Returns
        -------
        x, f, d     : tuple
                      x -- position of the minimum
                      f -- value of func at the minimum
                      d -- dictionary of information from routine
                            d['warnflag'] is
                                0 if converged
                                1 if too many function evaluations
                                2 if stopped for another reason, given in d['task']
                            d['grad'] is the gradient at the minimum (should be 0 ish)
                            d['funcalls'] is the number of function calls made
        """
           
        lambdaX = op.fmin_l_bfgs_b(self._gm_gmswls,[0.0, 0.0],args=[moments],approx_grad=True,bounds=[(-1.0,1.0), (0, None)])
        return lambdaX

    def _gm_gmswls(self, lambdapar, moments):
        """
        Preparation of moments for minimization in a GMSWLS framework
        """
        par=np.array([[float(lambdapar[0]),float(lambdapar[0])**2., lambdapar[1]]]).T
        vv = np.dot(moments[0], par)
        vv = vv - moments[1]
        return sum(vv**2)

def get_A1(S):
    """
    Builds A1 as in Arraiz et al [1]_

    .. math::

        A_1 = W' W - diag(w'_{.i} w_{.i})

    ...

    Parameters
    ----------

    S               : csr_matrix
                      PySAL W object converted into Scipy sparse matrix

    Returns
    -------

    Implicit        : csr_matrix
                      A1 matrix in scipy sparse format
    References
    ----------

    .. [1] Arraiz, I., Drukker, D. M., Kelejian, H., Prucha, I. R. (2010) "A
    Spatial Cliff-Ord-Type Model with Heteroskedastic Innovations: Small and
    Large Sample Results". Journal of Regional Science, Vol. 60, No. 2, pp.
    592-614.

           
    """
    StS = S.T * S
    d = SP.spdiags([StS.diagonal()], [0], S.get_shape()[0], S.get_shape()[1])
    d = d.asformat('csr')
    return StS - d

def get_spFilter(w,lamb,sf):
    '''
    computer the spatially filtered variables
    
    Parameters
    ----------
    w       : weight
              PySAL weights instance  
    lamb    : double
              spatial autoregressive parameter
    sf      : array
              the variable needed to compute the filter
    Returns
    --------
    rs      : array
              spatially filtered variable
    
    Examples
    --------

    >>> import numpy as np
    >>> import pysal
    >>> db=pysal.open("examples/columbus.dbf","r")
    >>> y = np.array(db.by_col("CRIME"))
    >>> y = np.reshape(y, (49,1))
    >>> w=pysal.open("examples/columbus.gal").read()        
    >>> solu = get_spFilter(w,0.5,y)
    >>> print solu
    [[  -8.9882875]
     [ -20.5685065]
     [ -28.196721 ]
     [ -36.9051915]
     [-111.1298   ]
     [ -14.5570555]
     [ -99.278625 ]
     [ -86.0715345]
     [-117.275209 ]
     [ -16.655933 ]
     [ -62.3681695]
     [ -73.0446045]
     [ -29.471835 ]
     [ -71.3954825]
     [-101.7297645]
     [-154.623178 ]
     [   9.732206 ]
     [ -58.3998535]
     [ -15.1412795]
     [-162.0080105]
     [ -25.5078975]
     [ -74.007205 ]
     [  -8.0705775]
     [-153.8715795]
     [-138.5858265]
     [-104.918187 ]
     [ -13.6139665]
     [-156.4892505]
     [-120.5168695]
     [ -52.541277 ]
     [ -11.0130095]
     [  -8.563781 ]
     [ -32.5883695]
     [ -20.300339 ]
     [ -76.6698755]
     [ -32.581708 ]
     [-110.5375805]
     [ -77.2471795]
     [  -5.1557885]
     [ -36.3949255]
     [ -12.69973  ]
     [  -2.647902 ]
     [ -71.81993  ]
     [ -63.405917 ]
     [ -35.1192345]
     [  -0.1726765]
     [  10.2496385]
     [ -30.452661 ]
     [ -18.2765175]]

    '''        
    # convert w into sparse matrix      
    w_matrix = w.sparse
    rs = sf - lamb * (w_matrix * sf)    
    
    return rs

def _test():
    import doctest
    doctest.testmod()

if __name__ == '__main__':

    _test()

    dbf = pysal.open('examples/columbus.dbf','r')
    y = np.array([dbf.by_col('HOVAL')]).T
    x = np.array([dbf.by_col('INC'), dbf.by_col('CRIME')]).T
    w = pysal.open('examples/columbus.gal', 'r').read()
    w.transform='r' #Needed to match R

    """
    print '\n\tGMSWLS model Example'
    model = GMSWLS(x, y, w)
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

