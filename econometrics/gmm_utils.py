"""
Tools for different GMM procedure estimations
"""

import numpy as np
import pylab as pl
from scipy import sparse as SP
import scipy.optimize as op
import numpy.linalg as la


def get_spCO(z,w,lambdaX):
    """
    Spatial Cochrane-Orcut Transf

    ...

    Parameters
    ----------

    Returns
    -------

    """
    return z - lambdaX * (w.sparse * z)

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

# copy from spHetErr.py
def get_S(w):
    """
    Converts pysal W to scipy csr_matrix
    ...

    Parameters
    ----------

    w               : W
                     Spatial weights instance

    Returns
    -------

    Implicit        : csr_matrix
                      PySAL W object converted into Scipy sparse matrix
                
    """
    data = []
    indptr = [0]
    indices = []
    for ob in w.id_order:
        data.extend(w.weights[ob])
        indptr.append(indptr[-1] + len(w.weights[ob]))
        indices.extend(w.neighbors[ob])
    data = np.array(data)
    indices = np.array(indices)
    indptr = np.array(indptr)
        
    return SP.csr_matrix((data,indices,indptr),shape=(w.n,w.n))

def get_spFilter(w,lamb,sf):
    '''
    computer the spatially filtered variables
    
    Parameters
    ----------
    w       : weight
              PySAL weights instance  
    lamda   : double
              spatial autoregressive parameter
    sp      : array
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
    >>> print solu.ys 
    >>>
    [[ -15.7812905]
    [ -32.158758 ]
    [ -32.4865705]
    [ -27.410798 ]
    [ -97.3822125]
    [  -3.9670885]
    [ -91.2635635]
    [ -99.43012  ]
    [-118.107001 ]
    [ -35.0421145]
    [ -35.2276465]
    [ -77.3782495]
    [ -35.4055665]
    [ -59.8451955]
    [ -75.117567 ]
    [-140.013057 ]
    [ -33.4549225]
    [ -29.5261305]
    [ -27.6813485]
    [-177.105114 ]
    [ -19.001647 ]
    [-119.702461 ]
    [ -42.9538575]
    [ -88.131421 ]
    [-132.2616395]
    [ -98.780399 ]
    [ -19.629096 ]
    [-118.43816  ]
    [ -92.400958 ]
    [ -60.3964295]
    [ -31.2535   ]
    [ -37.7388985]
    [ -44.0567675]
    [ -42.005636 ]
    [ -69.722601 ]
    [ -55.2186525]
    [ -62.0116055]
    [ -72.6139295]
    [ -25.6059015]
    [ -53.5011755]
    [ -24.8541415]
    [ -24.3181745]
    [ -40.9453225]
    [ -14.9928235]
    [ -22.078858 ]
    [ -12.8126545]
    [  10.124343 ]
    [  -9.115376 ]
    [ -11.508765 ]]
    '''        
    # convert w into sparse matrix      
    w_matrix = get_S(w)
    rs = sf - lamb * w_matrix.dot(sf,)    
    
    return rs
        

if __name__ == "__main__":
    
    import random
    import pysal
    from spHetErr import get_A1
    
    w=pysal.weights.lat2W(7,7, rook=False)
    w.transform='r'
    w.A1 = get_A1(w.sparse)
    random.seed(100)
    np.random.seed(100)
    u=np.random.normal(0,1,(w.n,1))
    u = np.random.randn(w.n,1) * (200*np.random.randn(w.n,1))

    m=moments_het(w,u)
    vc = get_vc_het(w, u, 0.1)
   

