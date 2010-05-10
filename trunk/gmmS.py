
import numpy as np
import pylab as pl
from scipy import sparse as SP

class Moments:
    """
    Class to compute all six components of the system of equations for a
    spatial error model with heteroskedasticity estimated by GMM

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
 
    """
    def __init__(self,w,u):

        ut = u.T
        S = w.S
        St = S.T

        utSt = ut * St
        A1u = w.A1 * u
        Su = S * u

        g1 = np.dot(ut, A1u)
        g2 = np.dot(ut, Su)
        self.g = np.array([[g1][0][0],[g2][0][0]]) / w.n

        G11 = 2 * (np.dot(utSt, A1u)) 
        G12 = -np.dot(utSt * w.A1, Su)
        G21 = np.dot(utSt, ((S + St) * u))
        G22 = -np.dot(utSt, (S * Su))
        self.G = np.array([[G11[0][0],G12[0][0]],[G21[0][0],G22[0][0]]]) / w.n

def get_vc(w, u, l):
    """
    Computes the VC matrix Psi based on lambda:

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

    u           : array
                  Residuals. nx1 array assumed to be aligned with w

    l           : float
                  Lambda parameter estimate
 
    Returns
    -------

    Implicit    : array
                  2x2 array with estimator of the variance-covariance matrix

    """
    e = (u - l * (w.S * u)) ** 2
    E = SP.lil_matrix(w.S.get_shape())
    E.setdiag(e.flat)
    E = E.asformat('csr')
    A1t = w.A1.T
    wt = w.S.T

    aPatE = (w.A1 + A1t) * E
    wPwtE = (w.S + wt) * E

    psi11 = aPatE * aPatE
    psi12 = aPatE * wPwtE
    psi22 = wPwtE * wPwtE 
    psi = map(np.sum, [psi11.diagonal(), psi12.diagonal(), psi22.diagonal()])
    return np.array([[psi[0], psi[1]], [psi[1], psi[2]]]) / (2 * w.n)

def get_vcF(w, u, l):
    """
    Same as get_vc() but implemented with full numpy arrays
    """
    e = (u - l * np.dot(w.full()[0], u)) ** 2
    E = np.eye(w.n) * e
    A1 = np.dot(w.full()[0].T, w.full()[0])
    for i in range(w.n):
        A1[i,i] = 0.

    aPatE = np.dot((A1 + A1.T), E)
    wPwtE = np.dot((w.full()[0] + w.full()[0].T), E)

    psi11 = np.dot(aPatE, aPatE)
    psi12 = np.dot(aPatE, wPwtE)
    psi22 = np.dot(wPwtE, wPwtE)
    psi = map(np.sum, [psi11.diagonal(), psi12.diagonal(), psi22.diagonal()])
    return np.array([[psi[0], psi[1]], [psi[1], psi[2]]]) / (2 * w.n)

if __name__ == "__main__":

    import random
    import pysal
    from spHetErr import get_S, get_A1
    w=pysal.weights.lat2W(7,7, rook=False)
    w.transform='r'
    w.S = get_S(w)
    w.A1 = get_A1(w.S)
    random.seed(100)
    np.random.seed(100)
    u=np.random.normal(0,1,(w.n,1))
    u = np.random.randn(w.n,1) * (200*np.random.randn(w.n,1))

    m=Moments(w,u)
    vc = get_vc(w, u, 0.1)
    vcF = get_vcF(w, u, 0.1)

