"""
Tools for different GMM procedure estimations
"""

import numpy as np
import pylab as pl
from scipy import sparse as SP
import scipy.optimize as op
import numpy.linalg as la
import twosls as tw

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
    A1 = get_A1(S)

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
    A1=get_A1(w.sparse)
    A1t = A1.T
    wt = w.sparse.T

    aPatE = (A1 + A1t) * E
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
    >>> reg = tw.TSLS_dev(y, X, yd, q)
    >>> w = pysal.rook_from_shapefile("examples/columbus.shp")
    >>> print get_a1a2(w, reg, 0.1)
    [array([[ 195.25744009],
       [ 134.95048367],
       [ 144.86497532],
       [ 197.14667952],
       [  52.33214022],
       [  81.69350113],
       [  21.57110524],
       [  39.25092151],
       [   6.4176049 ],
       [ 170.67203551],
       [ -60.15087711],
       [-100.19569464],
       [ -55.52837813],
       [-137.04929407],
       [-117.22065128],
       [-237.49411824],
       [ 225.3706287 ],
       [-203.42825553],
       [-174.85693576],
       [  -1.39376802],
       [-176.27654673],
       [ -50.67791491],
       [ 172.45987849],
       [-369.59572667],
       [-255.94408153],
       [-124.95298426],
       [ -56.13432618],
       [-163.02707821],
       [-254.6118325 ],
       [-342.08042841],
       [  87.27426796],
       [ 195.52015989],
       [ -48.61295331],
       [  25.43591635],
       [ -77.97103239],
       [ 161.58386522],
       [-292.86815848],
       [-163.35572807],
       [ 263.76071154],
       [  93.15017421],
       [ 229.06891219],
       [  24.8590105 ],
       [-149.51313393],
       [ -71.43398283],
       [-121.99742651],
       [ 242.30820653],
       [ 255.65175765],
       [ -67.73661086],
       [ -43.87175237]]), array([[-21.72029625],
       [ -7.31129615],
       [-25.52293215],
       [-71.65750555],
       [-28.66735051],
       [-12.63717599],
       [-28.68521374],
       [-29.58550686],
       [  2.96599179],
       [-30.93023241],
       [-17.80630999],
       [ -5.06459589],
       [-10.02397569],
       [  2.69963562],
       [  3.63990154],
       [ 14.25350242],
       [-50.38630826],
       [ 30.94803195],
       [ 23.88560703],
       [ 61.52079852],
       [ 25.35869067],
       [  3.59103615],
       [ -5.46861869],
       [ 66.63838572],
       [ 24.62544213],
       [  1.64026229],
       [  3.1368649 ],
       [  6.07269902],
       [ 30.61753457],
       [ 66.34539265],
       [ -6.5092134 ],
       [ -6.96971366],
       [ -0.47489675],
       [  3.33299607],
       [ 14.09521629],
       [-10.39550428],
       [ 65.41009544],
       [ 21.87671963],
       [-37.6785381 ],
       [ 43.75115273],
       [-11.03552002],
       [ 33.51704649],
       [ 28.45201281],
       [ 23.820111  ],
       [ 26.42182744],
       [-33.64750984],
       [-28.58150944],
       [  9.15976479],
       [ 23.67941368]])]

    """        
    zst = get_spFilter(w,lambdapar, reg.z).T
    us = get_spFilter(w,lambdapar, reg.u)
    alpha1 = (-2.0/w.n) * (np.dot((zst * get_A1(w.sparse)), us))
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
    >>> w=pysal.open("examples/columbus.GAL").read()        
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
  
def get_J(w, lamb, u):
    G=moments_het(w, u)[0]    
    J=np.dot(G, np.array([[1], [2*lamb]]))
    return J

def get_Omega_2SLS(w, lamb, reg):
    """
    Computes the variance-covariance matrix for 2SLS:
    ...

    Parameters
    ----------

    w           : W
                  Spatial weights instance 

    reg         : TSLS
                  Two stage least quare regression instance
                  
    lamb        : float
                  Spatial autoregressive parameter
 
    Returns
    -------

    omega       : array
                  (k+1)x(k+1)
                  
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
    >>> reg = TSLS(y, X, yd, q)
    >>> w = pysal.rook_from_shapefile("examples/columbus.shp")
    >>> print get_Omega_2SLS(w, 0.1, reg)
    >>>
    [[  2.02523835e+04   1.17710247e+03  -9.69212937e+02   1.28150689e+01]
    [  1.17710247e+03   1.14640082e+02  -7.61825004e+01   1.39419338e+00]
    [ -9.69212937e+02  -7.61825004e+01   5.56502280e+01  -8.80686177e-01]
    [  1.28150689e+01   1.39419338e+00  -8.80686177e-01   1.00264668e-01]]

    """

    pe = 0
    psi_dd_1=0
    psi_dd_2=0
    while (lamb**pe > 1e-10):
        psi_dd_1 = psi_dd_1 + (lamb**pe) * reg.h.T * (w.sparse**pe)
        psi_dd_2 = psi_dd_2 + (lamb**pe) * (w.sparse.T**pe) * reg.h
        pe = pe + 1
        
    sigma=get_psi_sigma(w, reg.u, lamb)
    psi_dd_h=(1.0/w.n)*psi_dd_1*sigma
    psi_dd=np.dot(psi_dd_h, psi_dd_2)
    a1a2=get_a1a2(w, reg, lamb)
    psi_dl=np.dot(psi_dd_h, np.hstack((a1a2[0],a1a2[1])))
    psi=get_vc_het(w,sigma)    
    psii=la.inv(psi)
    psi_o=np.hstack((np.vstack((psi_dd, psi_dl.T)), np.vstack((psi_dl, psi))))
    J=get_J(w, lamb, reg.u)
    jtpsii=np.dot(J.T, psii)
    jtpsiij=np.dot(jtpsii, J)
    jtpsiiji=la.inv(jtpsiij)
    omega_1=np.dot(jtpsiiji, jtpsii)
    omega_2=np.dot(np.dot(psii, J), jtpsiiji)
    om_1_s=omega_1.shape
    om_2_s=omega_2.shape
    p_s=reg.pfora1a2.shape
    omega_left=np.hstack((np.vstack((reg.pfora1a2.T, np.zeros((om_1_s[0],p_s[0])))), 
               np.vstack((np.zeros((p_s[1], om_1_s[1])), omega_1))))
    omega_right=np.hstack((np.vstack((reg.pfora1a2, np.zeros((om_2_s[0],p_s[1])))), 
               np.vstack((np.zeros((p_s[0], om_2_s[1])), omega_2))))
    omega=np.dot(np.dot(omega_left, psi_o), omega_right)
    
    return omega
    
                  
    

def _test():
    import doctest
    doctest.testmod()

if __name__ == '__main__':
    _test()
'''
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
    import numpy as np
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
    reg = tw.TSLS_dev(y, X, yd, q)
    w = pysal.rook_from_shapefile("examples/columbus.shp")
    print get_Omega_2SLS(w, 0.1, reg)
    #print get_a1a2(w, reg, 0.1)
    
    

