"""
Tools for different GMM procedure estimations
"""

import numpy as np
import pylab as pl
from scipy import sparse as SP
import scipy.optimize as op
import numpy.linalg as la
import twosls as tw
import gstwosls as gst

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

def optim_moments(moments, vcX=np.array([0])):
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
    if moments[0].shape[0] == 2:
        optim_par = lambda par: foptim_par(np.array([[float(par[0]),float(par[0])**2.]]).T,moments)
        start = [0.0]
        bounds=[(-1.0,1.0)]
    if moments[0].shape[0] == 3:
        optim_par = lambda par: foptim_par(np.array([[float(par[0]),float(par[0])**2.,par[1]]]).T,moments)
        start = [0.0,0.0]
        bounds=[(-1.0,1.0),(0,None)]        
    lambdaX = op.fmin_l_bfgs_b(optim_par,start,approx_grad=True,bounds=bounds)
    return lambdaX[0][0]

def foptim_par(par,moments):
    """ 
    Preparation of the function of moments for minimization
    ...

    Parameters
    ----------

    lambdapar       : float
                      Spatial autoregressive parameter
    moments         : list
                      List of Moments with G (moments[0]) and g (moments[1])

    Returns
    -------

    minimum         : float
                      sum of square residuals (e) of the equation system 
                      moments.g - moments.G * lambdapar = e
    """
    vv=np.dot(moments[0],par)
    vv2=vv-moments[1]
    return sum(vv2**2)

def get_spFilter(w,lamb,sf):
    '''
    Compute the spatially filtered variables
    
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

'''
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
    >>> reg = tw.TSLS_dev(y, X, yd, q)
    >>> w = pysal.rook_from_shapefile("examples/columbus.shp")
    >>> print get_Omega_2SLS(w, 0.1, reg)
    [[  1.61856494e+05  -1.69448764e+03  -2.67133097e+03  -3.61811953e+02]
     [ -1.69448764e+03   1.78247578e+01   2.78845560e+01   3.80403384e+00]
     [ -2.67133097e+03   2.78845560e+01   4.42444411e+01   5.94374825e+00]
     [ -3.61811953e+02   3.80403384e+00   5.94374825e+00   7.73788069e-02]]

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
'''

def _test():
    import doctest
    doctest.testmod()

if __name__ == '__main__':
    _test() 
