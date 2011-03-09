import numpy as np
import gmm_utils as GMM
from gmm_utils import get_A1, get_spFilter
import pysal.spreg.ols as OLS
import twosls as TSLS
from scipy import sparse as SP

class SWLS_Het:
    """
    GMM method for a spatial error model with heteroskedasticity

    Based on Arraiz et al [1]_

    ...

    Parameters
    ----------

    x           : array
                  nxk array with independent variables aligned with y
    y           : array
                  nx1 array with dependent variables
    w           : W
                  PySAL weights instance aligned with y and with instances S
                  and A1 created
    cycles      : int
                  Optional. Number of iterations of steps 2a. and 2b. Set to 1
                  by default

    Attributes
    ----------
    
    betas       : array
                  (k+1)x1 array with estimates for betas and lambda
    u           : array
                  nx1 array with residuals
    vm          : array
                  (k+1)x(k+1) variance-covariance matrix

    References
    ----------

    .. [1] Arraiz, I., Drukker, D. M., Kelejian, H., Prucha, I. R. (2010) "A
    Spatial Cliff-Ord-Type Model with Heteroskedastic Innovations: Small and
    Large Sample Results". Journal of Regional Science, Vol. 60, No. 2, pp.
    592-614.

    Examples
    --------
    >>> import numpy as np
    >>> from testing_utils import Test_Data as DAT
    >>> data = DAT()
    >>> y, x, w = data.y, data.x, data.w
    >>> reg = SWLS_Het(y, x, w)
    >>> print np.hstack((reg.betas,np.sqrt(reg.vm.diagonal()).reshape(7,1)))
    [[ 0.08028647  0.09883845]
     [-0.17811755  0.10271992]
     [-0.05991007  0.12766904]
     [-0.01487806  0.10826346]
     [ 0.09024685  0.09246069]
     [ 0.14459452  0.10005024]
     [ 0.00068073  0.08663456]]
    """

    def __init__(self,y,x,w,cycles=1,constant=True): ######Inserted i parameter here for iterations...
        #1a. OLS --> \tilde{betas}
        ols = OLS.BaseOLS(y, x,constant)

        #1b. GMM --> \tilde{\lambda1}
        moments = GMM.moments_het(w, ols.u)
        lambda1 = GMM.optimizer_het(moments)[0][0]

        #1c. GMM --> \tilde{\lambda2}
        sigma = GMM.get_psi_sigma(w, ols.u, lambda1)
        vc1 = GMM.get_vc_het(w, sigma)
        lambda2 = GMM.optimizer_het(moments,vc1)[0][0]
        
        ols.betas, lambda3, ols.u, vc2, G, ols.predy = self.iterate(cycles,ols,w,lambda2)
        #Output
        self.betas = np.vstack((ols.betas,lambda3))
        self.vm = GMM.get_vm_het(G,lambda3,ols,w,vc2)
        self.u = ols.u

    def iterate(self,cycles,reg,w,lambda2):

        for n in range(cycles): #### Added loop.
            #2a. reg -->\hat{betas}
            xs,ys = get_spFilter(w,lambda2,reg.x),get_spFilter(w,lambda2,reg.y)            
            beta_i = np.dot(np.linalg.inv(np.dot(xs.T,xs)),np.dot(xs.T,ys))
            predy = np.dot(reg.x, beta_i)
            u = reg.y - predy
            #2b. GMM --> \hat{\lambda}
            moments_i = GMM.moments_het(w, u)
            sigma_i =  GMM.get_psi_sigma(w, u, lambda2)
            vc2 = GMM.get_vc_het(w, sigma_i)
            lambda2 = GMM.optimizer_het(moments_i,vc2)[0][0]
        return beta_i,lambda2,u,vc2,moments_i[0], predy

class GSTSLS_Het:
    """
    GMM method for a spatial error model with heteroskedasticity and endogenous variables

    Based on Arraiz et al [1]_

    ...

    Parameters
    ----------

    x           : array
                  nxk array with independent variables aligned with y
    y           : array
                  nx1 array with dependent variables
    yend        : array
                  non-spatial endogenous variables
    q           : array
                  array of instruments for yend (note: this should not contain
                  any variables from x; spatial instruments are computed by 
                  default)
    w           : W
                  PySAL weights instance aligned with y and with instances S
                  and A1 created
    cycles      : int
                  Optional. Number of iterations of steps 2a. and 2b. Set to 1
                  by default

    Attributes
    ----------
    
    betas       : array
                  (k+1)x1 array with estimates for betas and lambda
    u           : array
                  nx1 array with residuals
    vm          : array
                  (k+1)x(k+1) variance-covariance matrix

    References
    ----------

    .. [1] Arraiz, I., Drukker, D. M., Kelejian, H., Prucha, I. R. (2010) "A
    Spatial Cliff-Ord-Type Model with Heteroskedastic Innovations: Small and
    Large Sample Results". Journal of Regional Science, Vol. 60, No. 2, pp.
    592-614.

    Examples
    --------
    >>> import numpy as np
    >>> import pysal
    >>> db=pysal.open("examples/columbus.dbf","r")
    >>> y = np.array(db.by_col("CRIME"))
    >>> y = np.reshape(y, (49,1))
    >>> X = []
    >>> X.append(db.by_col("INC"))
    >>> X = np.array(X).T
    >>> yd = []
    >>> yd.append(db.by_col("HOVAL"))
    >>> yd = np.array(yd).T
    >>> q = []
    >>> q.append(db.by_col("DISCBD"))
    >>> q = np.array(q).T
    >>> w = pysal.rook_from_shapefile("examples/columbus.shp")
    >>> w.transform = 'r'
    >>> reg = GSTSLS_Het(y, X, yd, q, w)
    >>> print np.hstack((reg.betas,np.sqrt(reg.vm.diagonal()).reshape(4,1)))
    [[  8.22535188e+01   9.99563735e+01]
     [  9.18288506e-01   9.30827948e+00]
     [ -1.56407478e+00   6.07134267e+00]
     [  5.02913162e-01   8.12910889e+02]]
    
    """

    def __init__(self,y,x,yend,q,w,cycles=1,constant=True):
        #1a. OLS --> \tilde{betas}
        tsls = TSLS.BaseTSLS(y, x, yend, q, constant)

        #1b. GMM --> \tilde{\lambda1}
        moments = GMM.moments_het(w, tsls.u)
        lambda1 = GMM.optimizer_het(moments)[0][0]

        #1c. GMM --> \tilde{\lambda2}
        vc1 = GMM.get_vc_het_tsls(w, tsls, lambda1)
        lambda2 = GMM.optimizer_het(moments,vc1)[0][0]
        
        tsls.betas, lambda3, tsls.u, vc2, G, tsls.predy = self.iterate(cycles,tsls,w,lambda2)
        self.u = tsls.u
        #Output
        self.betas = np.vstack((tsls.betas,lambda3))
        self.vm = GMM.get_Omega_GS2SLS(w, lambda3, tsls, G, vc2)

    def iterate(self,cycles,reg,w,lambda2):

        for n in range(cycles):
            #2a. reg -->\hat{betas}
            xs,ys = get_spFilter(w,lambda2,reg.x),get_spFilter(w,lambda2,reg.y)
            yend_s = get_spFilter(w,lambda2, reg.yend)
            tsls = TSLS.BaseTSLS(ys, xs, yend_s, reg.q, constant=False)
            predy = np.dot(np.hstack((reg.x, reg.yend)), tsls.betas)
            tsls.u = reg.y - predy
            #2b. GMM --> \hat{\lambda}
            moments_i = GMM.moments_het(w, tsls.u)
            vc2 = GMM.get_vc_het_tsls(w, tsls, lambda2)
            lambda2 = GMM.optimizer_het(moments_i,vc2)[0][0]
        return tsls.betas,lambda2,tsls.u,vc2,moments_i[0], predy


def _test():
    import doctest
    doctest.testmod()

if __name__ == '__main__':
    _test()
    import numpy as np
    import pysal
    db=pysal.open("examples/columbus.dbf","r")
    y = np.array(db.by_col("CRIME"))
    y = np.reshape(y, (49,1))
    X = []
    X.append(db.by_col("INC"))
    X = np.array(X).T
    yd = []
    yd.append(db.by_col("HOVAL"))
    yd = np.array(yd).T
    q = []
    q.append(db.by_col("DISCBD"))
    q = np.array(q).T
    w = pysal.rook_from_shapefile("examples/columbus.shp")
    w.transform = 'r'
    reg = GSTSLS_Het(y, X, yd, q, w)
    print "Dependent variable: CRIME"
    print "Variable  Coef.  S.E."
    print "Constant %5.4f %5.4f" % (reg.betas[0],np.sqrt(reg.vm.diagonal())[0])
    for i in range(X.shape[1]+1):
        print "Var_%s %5.4f %5.4f" % (i+1,reg.betas[i+1],np.sqrt(reg.vm.diagonal())[i+1])
    print "Lambda: %5.4f %5.4f" % (reg.betas[-1],np.sqrt(reg.vm.diagonal())[-1])
