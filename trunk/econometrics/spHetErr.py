import numpy as np
import gmm_utils as GMM
import pysal.spreg.ols as OLS
import twosls as TSLS
from scipy import sparse as SP
import numpy.linalg as la

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
        moments = moments_het(w, ols.u)
        lambda1 = GMM.optim_moments(moments)

        #1c. GMM --> \tilde{\lambda2}
        sigma = get_psi_sigma(w, ols.u, lambda1)
        vc1 = get_vc_het(w, sigma)
        lambda2 = GMM.optim_moments(moments,vc1)
        
        ols.betas, lambda3, ols.u, vc2, G, ols.predy = self.iterate(cycles,ols,w,lambda2)
        #Output
        self.betas = np.vstack((ols.betas,lambda3))
        self.vm = get_vm_het(G,lambda3,ols,w,vc2)
        self.u = ols.u

    def iterate(self,cycles,reg,w,lambda2):
        for n in range(cycles):
            #2a. reg -->\hat{betas}
            xs,ys = GMM.get_spFilter(w,lambda2,reg.x),GMM.get_spFilter(w,lambda2,reg.y)            
            beta_i = np.dot(np.linalg.inv(np.dot(xs.T,xs)),np.dot(xs.T,ys))
            predy = np.dot(reg.x, beta_i)
            u = reg.y - predy
            #2b. GMM --> \hat{\lambda}
            moments_i = moments_het(w, u)
            sigma_i =  get_psi_sigma(w, u, lambda2)
            vc2 = get_vc_het(w, sigma_i)
            lambda2 = GMM.optim_moments(moments_i,vc2)
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
    >>> reg = GSTSLS_Het(y, X, w, yd, q)
    >>> print np.around(np.hstack((reg.betas,np.sqrt(reg.vm.diagonal()).reshape(4,1))),4)
    [[  8.25947000e+01   9.54856000e+01]
     [  6.94700000e-01   8.92810000e+00]
     [ -1.49070000e+00   5.79200000e+00]
     [  4.85700000e-01   9.60023000e+02]]
    
    """

    def __init__(self,y,x,w,yend,q,cycles=1,constant=True):
        #1a. OLS --> \tilde{betas}
        tsls = TSLS.BaseTSLS(y, x, yend, q=q, constant=constant)

        #1b. GMM --> \tilde{\lambda1}
        moments = moments_het(w, tsls.u)
        lambda1 = GMM.optim_moments(moments)

        #1c. GMM --> \tilde{\lambda2}
        vc1 = get_vc_het_tsls(w, tsls, lambda1)
        lambda2 = GMM.optim_moments(moments,vc1)
        
        tsls.betas, lambda3, tsls.u, vc2, G, tsls.predy = self.iterate(cycles,tsls,w,lambda2)
        self.u = tsls.u
        #Output
        self.betas = np.vstack((tsls.betas,lambda3))
        self.vm = get_Omega_GS2SLS(w, lambda3, tsls, G, vc2)

    def iterate(self,cycles,reg,w,lambda2):
        for n in range(cycles):
            #2a. reg -->\hat{betas}
            xs,ys = GMM.get_spFilter(w,lambda2,reg.x),GMM.get_spFilter(w,lambda2,reg.y)
            yend_s = GMM.get_spFilter(w,lambda2, reg.yend)
            tsls = TSLS.BaseTSLS(ys, xs, yend_s, h=reg.h, constant=False)
            predy = np.dot(np.hstack((reg.x, reg.yend)), tsls.betas)
            tsls.u = reg.y - predy
            #2b. GMM --> \hat{\lambda}
            moments_i = moments_het(w, tsls.u)
            vc2 = get_vc_het_tsls(w, tsls, lambda2)
            lambda2 = GMM.optim_moments(moments_i,vc2)
        return tsls.betas,lambda2,tsls.u,vc2,moments_i[0], predy

class GSTSLS_Het_lag(GSTSLS_Het):
    '''
    Version of GSTSLS_Het with spatial endogenous var
    '''
    def __init__(self, y, x, w, yend=None, q=None, w_lags=1,\
                    constant=True, robust=None, cycles=1):
        # Create spatial lag of y
        yl = pysal.lag_spatial(w, y)
        if issubclass(type(yend), np.ndarray):  # spatial and non-spatial instruments
            lag_vars = np.hstack((x, q))
            spatial_inst = self.get_lags(lag_vars, w, w_lags)
            q = np.hstack((q, spatial_inst))
            yend = np.hstack((yend, yl))
        elif yend == None:                   # spatial instruments only
            q = self.get_lags(x, w, w_lags)
            yend = yl
        else:
            raise Exception, "invalid value passed to yend"
        GSTSLS_Het.__init__(self, y, x, w, yend, q, cycles=cycles, constant=constant)

    def get_lags(self, x, w, w_lags):
        lag = pysal.lag_spatial(w, x)
        spat_inst = lag
        for i in range(w_lags-1):
            lag = pysal.lag_spatial(w, lag)
            spat_inst = np.hstack((spat_inst, lag))
        return spat_inst


def moments_het(w, u):
    """
    Class to compute all six components of the system of equations for a
    spatial error model with heteroskedasticity estimated by GMM as in Arraiz
    et al [1]_

    Scipy sparse matrix version. It implements eqs. A.1 in Appendix A of
    Arraiz et al. (2007) by using all matrix manipulation
    
    [g1] + [G11 G12] *  [\lambda]    = [0]
    [g2]   [G21 G22]    [\lambda^2]    [0]

    NOTE: 'residuals' has been renamed 'u' to fit paper notation

    ...
    
    Parameters
    ----------

    w           : W
                  Spatial weights instance (requires 'S' and 'A1')

    u           : array
                  Residuals. nx1 array assumed to be aligned with w
 
    Attributes
    ----------

    moments     : list
                  List of two arrays corresponding to the matrices 'G' and
                  'g', respectively.

    References
    ----------

    .. [1] Arraiz, I., Drukker, D. M., Kelejian, H., Prucha, I. R. (2010) "A
    Spatial Cliff-Ord-Type Model with Heteroskedastic Innovations: Small and
    Large Sample Results". Journal of Regional Science, Vol. 60, No. 2, pp.
    592-614.

    """
    ut = u.T
    S = w.sparse
    St = S.T
    A1 = GMM.get_A1(S)

    utSt = ut * St
    A1u = A1 * u
    Su = S * u

    g1 = np.dot(ut, A1u)
    g2 = np.dot(ut, Su)
    g = np.array([[g1][0][0],[g2][0][0]]) / w.n

    G11 = 2 * (np.dot(utSt, A1u)) 
    G12 = -np.dot(utSt * A1, Su)
    G21 = np.dot(utSt, ((S + St) * u))
    G22 = -np.dot(utSt, (S * Su))
    G = np.array([[G11[0][0],G12[0][0]],[G21[0][0],G22[0][0]]]) / w.n
    return [G, g]

def get_psi_sigma(w, u, l):
    """
    Computes the Sigma matrix needed to compute Psi

    Parameters
    ----------
    w           : W
                  Spatial weights instance (requires 'S' and 'A1')

    u           : array
                  nx1 vector of residuals

    l           : float
                  Lambda

    """

    e = (u - l * (w.sparse * u)) ** 2
    E = SP.lil_matrix(w.sparse.get_shape())
    E.setdiag(e.flat)
    E = E.asformat('csr')
    return E

def get_vc_het(w, E):
    """
    Computes the VC matrix Psi based on lambda as in Arraiz et al [1]_:

    ..math::

        \tilde{Psi} = \left(\begin{array}{c c}
                            \psi_{11} & \psi_{12} \\
                            \psi_{21} & \psi_{22} \\
                      \end{array} \right)

    NOTE: psi12=psi21

    ...

    Parameters
    ----------

    w           : W
                  Spatial weights instance (requires 'S' and 'A1')

    E           : sparse matrix
                  Sigma
 
    Returns
    -------

    Psi         : array
                  2x2 array with estimator of the variance-covariance matrix

    References
    ----------

    .. [1] Arraiz, I., Drukker, D. M., Kelejian, H., Prucha, I. R. (2010) "A
    Spatial Cliff-Ord-Type Model with Heteroskedastic Innovations: Small and
    Large Sample Results". Journal of Regional Science, Vol. 60, No. 2, pp.
    592-614.

    """
    A1=GMM.get_A1(w.sparse)
    A1t = A1.T
    wt = w.sparse.T

    aPatE = (A1 + A1t) * E
    wPwtE = (w.sparse + wt) * E

    psi11 = aPatE * aPatE
    psi12 = aPatE * wPwtE
    psi22 = wPwtE * wPwtE 
    psi = map(np.sum, [psi11.diagonal(), psi12.diagonal(), psi22.diagonal()])
    return np.array([[psi[0], psi[1]], [psi[1], psi[2]]]) / (2 * w.n)

def get_vm_het(G, lamb, reg, w, psi):
    """
    Computes the variance-covariance matrix Omega as in Arraiz et al [1]_:
    ...

    Parameters
    ----------

    G           : array
                  G from moments equations

    lamb        : float
                  Final lambda from spHetErr estimation

    reg         : regression object
                  output instance from a regression model

    u           : array
                  nx1 vector of residuals

    w           : W
                  Spatial weights instance

    psi         : array
                  2x2 array with the variance-covariance matrix of the moment equations
 
    Returns
    -------

    vm          : array
                  (k+1)x(k+1) array with the variance-covariance matrix of the parameters

    References
    ----------

    .. [1] Arraiz, I., Drukker, D. M., Kelejian, H., Prucha, I. R. (2010) "A
    Spatial Cliff-Ord-Type Model with Heteroskedastic Innovations: Small and
    Large Sample Results". Journal of Regional Science, Vol. 60, No. 2, pp.
    592-614.

    """

    J = np.dot(G, np.array([[1],[2 * lamb]]))
    Zs = GMM.get_spFilter(w,lamb,reg.x)
    ZstEZs = np.dot((Zs.T * get_psi_sigma(w, reg.u, lamb)), Zs)
    ZsZsi = la.inv(np.dot(Zs.T,Zs))
    omega11 = w.n * np.dot(np.dot(ZsZsi,ZstEZs),ZsZsi)
    omega22 = la.inv(np.dot(np.dot(J.T,la.inv(psi)),J))
    zero = np.zeros((reg.k,1),float)
    vm = np.vstack((np.hstack((omega11, zero)),np.hstack((zero.T, omega22)))) / w.n
    return vm

def get_vc_het_tsls(w, reg, lambdapar):

    sigma = get_psi_sigma(w, reg.u, lambdapar)
    vc1 = get_vc_het(w, sigma)
    a1, a2 = get_a1a2(w, reg, lambdapar)
    a1s = a1.T * sigma
    a2s = a2.T * sigma
    psi11 = float(np.dot(a1s, a1))
    psi12 = float(np.dot(a1s, a2))
    psi21 = float(np.dot(a2s, a1))
    psi22 = float(np.dot(a2s, a2))
    psi = np.array([[psi11, psi12], [psi21, psi22]]) / w.n
    return vc1 + psi

def get_Omega_GS2SLS(w, lamb, reg, G, psi):
    """
    Computes the variance-covariance matrix for GS2SLS:
    ...

    Parameters
    ----------

    w           : W
                  Spatial weights instance 

    lamb        : float
                  Spatial autoregressive parameter
                  
    reg         : GSTSLS
                  Generalized Spatial two stage least quare regression instance
    G           : array
                  Moments
    psi         : array
                  Weighting matrix
 
    Returns
    -------

    omega       : array
                  (k+1)x(k+1)                 
    """
    
    sigma=get_psi_sigma(w, reg.u, lamb)
    psi_dd_1=(1.0/w.n) * reg.h.T * sigma 
    psi_dd = np.dot(psi_dd_1, reg.h)
    a1a2=get_a1a2(w, reg, lamb)
    psi_dl=np.dot(psi_dd_1,np.hstack(tuple(a1a2)))
    psi_o=np.hstack((np.vstack((psi_dd, psi_dl.T)), np.vstack((psi_dl, psi))))
    psii=la.inv(psi)
   
    j = np.dot(G, np.array([[1.], [2*lamb]]))
    jtpsii=np.dot(j.T, psii)
    jtpsiij=np.dot(jtpsii, j)
    jtpsiiji=la.inv(jtpsiij)
    omega_1=np.dot(jtpsiiji, jtpsii)
    omega_2=np.dot(np.dot(psii, j), jtpsiiji)
    om_1_s=omega_1.shape
    om_2_s=omega_2.shape
    p_s=reg.pfora1a2.shape
    
    omega_left=np.hstack((np.vstack((reg.pfora1a2.T, np.zeros((om_1_s[0],p_s[0])))), 
               np.vstack((np.zeros((p_s[1], om_1_s[1])), omega_1))))
    omega_right=np.hstack((np.vstack((reg.pfora1a2, np.zeros((om_2_s[0],p_s[1])))), 
               np.vstack((np.zeros((p_s[0], om_2_s[1])), omega_2))))
    omega=np.dot(np.dot(omega_left, psi_o), omega_right)    
    return omega

def get_a1a2(w,reg,lambdapar):
    """
    Computes the a1 in psi equation:
    ...

    Parameters
    ----------

    w           : W
                  Spatial weights instance 

    reg         : TSLS
                  Two stage least quare regression instance
                  
    lambdapar   : float
                  Spatial autoregressive parameter
 
    Returns
    -------

    [a1, a2]    : list
                  a1 and a2 are two nx1 array in psi equation

    References
    ----------

    .. [1] Anselin, L. GMM Estimation of Spatial Error Autocorrelation with Heteroskedasticity
    
    """        
    zst = GMM.get_spFilter(w,lambdapar, reg.z).T
    us = GMM.get_spFilter(w,lambdapar, reg.u)
    alpha1 = (-2.0/w.n) * (np.dot((zst * GMM.get_A1(w.sparse)), us))
    alpha2 = (-1.0/w.n) * (np.dot((zst * (w.sparse + w.sparse.T)), us))
    v1t = np.dot(np.dot(reg.h, reg.pfora1a2), alpha1).T
    v2t = np.dot(np.dot(reg.h, reg.pfora1a2), alpha2).T
    a1t = 0
    a2t = 0
    pe = 0
    while (lambdapar**pe > 1e-10):
        a1t = a1t + (lambdapar**pe) * v1t * (w.sparse**pe)
        a2t = a2t + (lambdapar**pe) * v2t * (w.sparse**pe)
        pe = pe + 1
    return [a1t.T, a2t.T]

def _test():
    import doctest
    doctest.testmod()

if __name__ == '__main__':
    #_test()
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
    reg = SWLS_Het(y, X, w)
    print "Exogenous variables only:"
    print "Dependent variable: CRIME"
    print "Variable  Coef.  S.E."
    print "Constant %5.4f %5.4f" % (reg.betas[0],np.sqrt(reg.vm.diagonal())[0])
    for i in range(len(reg.betas)-2):
        print "Var_%s %5.4f %5.4f" % (i+1,reg.betas[i+1],np.sqrt(reg.vm.diagonal())[i+1])
    print "Lambda: %5.4f %5.4f" % (reg.betas[-1],np.sqrt(reg.vm.diagonal())[-1])
    print '$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$'
    print "Spatial Lag:"
    reg = GSTSLS_Het_lag(y, X, w, yd, q)
    print "Dependent variable: CRIME"
    print "Variable  Coef.  S.E."
    print "Constant %5.4f %5.4f" % (reg.betas[0],np.sqrt(reg.vm.diagonal())[0])
    for i in range(len(reg.betas)-2):
        print "Var_%s %5.4f %5.4f" % (i+1,reg.betas[i+1],np.sqrt(reg.vm.diagonal())[i+1])
    print "Lambda: %5.4f %5.4f" % (reg.betas[-1],np.sqrt(reg.vm.diagonal())[-1])
