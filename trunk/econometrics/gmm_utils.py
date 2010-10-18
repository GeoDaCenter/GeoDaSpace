"""
Tools for different GMM procedure estimations
"""

import numpy as np
import pylab as pl
from scipy import sparse as SP
import scipy.optimize as op
import numpy.linalg as la

# Arraiz et al.

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

    utSt = ut * St
    A1u = w.A1 * u
    Su = S * u

    g1 = np.dot(ut, A1u)
    g2 = np.dot(ut, Su)
    g = np.array([[g1][0][0],[g2][0][0]]) / w.n

    G11 = 2 * (np.dot(utSt, A1u)) 
    G12 = -np.dot(utSt * w.A1, Su)
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
    
    A1t = w.A1.T
    wt = w.sparse.T

    aPatE = (w.A1 + A1t) * E
    wPwtE = (w.sparse + wt) * E

    psi11 = aPatE * aPatE
    psi12 = aPatE * wPwtE
    psi22 = wPwtE * wPwtE 
    psi = map(np.sum, [psi11.diagonal(), psi12.diagonal(), psi22.diagonal()])
    return np.array([[psi[0], psi[1]], [psi[1], psi[2]]]) / (2 * w.n)

def optimizer_het(moments, vcX=np.array([0])):
    """
    Optimization of moments
    ...

    Parameters
    ----------

    moments     : Moments
                  Instance of gmm_utils.moments_het with G and g
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
    if vcX.any():
        Ec = np.transpose(la.cholesky(la.inv(vcX)))
        moments[0] = np.dot(Ec,moments[0])
        moments[1] = np.dot(Ec,moments[1])
        
    lambdaX = op.fmin_l_bfgs_b(gm_het,[0.0],args=[moments],approx_grad=True,bounds=[(-1.0,1.0)])
    return lambdaX

def gm_het(lambdapar,moments):
    """ 
    Preparation of moments for minimization as in Arraiz et al [1]_
    ...

    Parameters
    ----------

    lambdapar       : float
                      Spatial autoregressive parameter
    moments         : moments_het
                      Instance of gmm_utils.Moments with G and g

    Returns
    -------

    minimum         : float
                      sum of square residuals (e) of the equation system 
                      moments.g - moments.G * lambdapar = e

    References
    ----------

    .. [1] Arraiz, I., Drukker, D. M., Kelejian, H., Prucha, I. R. (2010) "A
    Spatial Cliff-Ord-Type Model with Heteroskedastic Innovations: Small and
    Large Sample Results". Journal of Regional Science, Vol. 60, No. 2, pp.
    592-614.


    """
    par=np.array([float(lambdapar[0]),float(lambdapar[0])**float(2)])
    vv=np.inner(moments[0],par)
    vv2=np.reshape(vv,[2,1])-moments[1]
    return sum(vv2**2)

def get_vm_het(G, lamb, reg, u, w, psi):
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
    Zs = get_spCO(reg.x,w,lamb)
    ZstEZs = np.dot((Zs.T * get_psi_sigma(w, u, lamb)), Zs)
    ZsZsi = la.inv(np.dot(Zs.T,Zs))
    omega11 = w.n * np.dot(np.dot(ZsZsi,ZstEZs),ZsZsi)
    omega22 = la.inv(np.dot(np.dot(J.T,la.inv(psi)),J))
    zero = np.zeros((reg.k,1),float)
    vm = np.vstack((np.hstack((omega11, zero)),np.hstack((zero.T, omega22)))) / w.n
    return vm

# GSLS

def momentsGSLS(w, u):

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

def optimizer_gsls(moments):
    """
    Optimization of moments
    ...

    Parameters
    ----------

    moments     : Moments
                  Instance of gmm_utils.Moments with G and g
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
       
    lambdaX = op.fmin_l_bfgs_b(gm_gsls,[0.0, 0.0],args=[moments],approx_grad=True,bounds=[(-1.0,1.0), (0, None)])
    return lambdaX

def gm_gsls(lambdapar, moments):
    """
    Preparation of moments for minimization in a GSLS framework
    """
    par=np.array([[float(lambdapar[0]),float(lambdapar[0])**2., lambdapar[1]]]).T
    vv = np.dot(moments[0], par)
    vv = vv - moments[1]
    return sum(vv**2)


def get_a1a2(w, h, u, z, lambdapar):
    """
    Computes the a1 in psi equation:
    ...

    Parameters
    ----------

    w           : W
                  Spatial weights instance 

    h           : matrix
                  nxp matrix of Instruments
                  
    u           : array
                  nx1 array of residuals
                  
    z           : matrix
                  nxk matrix of exogenous variables
                  
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
    
    hth = np.dot(h.T, h)
    htz = np.dot(h.T, z)
    zth = htz.T
    hthI = hth.I
    p1 = np.dot(hthI, htz)
    p2 = np.dot(((1.0/w.n) * zth), p1)
    p = np.dot(p1, p2.I)
    zst = get_spFilter(w,lambdapar, z).T
    us = get_spFilter(w,lambdapar, u)
    alpha1 = (-2.0/w.n) * (np.dot((zst * get_A1(w.sparse)), us))
    alpha2 = (-1.0/w.n) * (np.dot((zst * (w.sparse + w.sparse.T)), us))
    v1t = np.dot(np.dot(h, p), alpha1).T
    v2t = np.dot(np.dot(h, p), alpha2).T
    a1t = 0
    a2t = 0
    pe = 0
    while (lambdapar**pe > 1e-10):
        a1t = a1t + (lambdapar**pe) * v1t * (w.sparse**pe)
        a2t = a2t + (lambdapar**pe) * v2t * (w.sparse**pe)
        pe = pe + 1
    return [a1t.T, a2t.T]

def get_spFilter(w,lamb,sf):
    '''
    computer the spatially filtered variables
    
    Parameters
    ----------
    w       : weight
              PySAL weights instance  
    lamda   : double
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
    >>> w=pysal.open("examples/columbus.GAL").read()        
    >>> solu = get_spFilter(w,0.5,y)
    >>> print solu
    >>>
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
    rs = sf - lamb * w_matrix.dot(sf,)    
    
    return rs
        

if __name__ == "__main__":
    
    import random
    import pysal
    from spHetErr import get_A1
 
    #w=pysal.weights.lat2W(7,7, rook=False)
    #w.transform='r'
    #w.A1 = get_A1(w.sparse)
    #random.seed(100)
    #np.random.seed(100)
    #u=np.random.normal(0,1,(w.n,1))
    #u = np.random.randn(w.n,1) * (200*np.random.randn(w.n,1))

    #m=moments_het(w,u)
    #vc = get_vc_het(w, u, 0.1)
    w=pysal.weights.lat2W(2,2)
    z=np.matrix([[1,6], [2,3], [5,7], [4,9]])
    u=np.array([[3], [7], [8], [4]])
    zl=w.sparse * z
    h=np.hstack((z, zl))
    a1,a2=get_a1a2(w,h,u, z, 0.1)
    print a1,a2

