
from time import time
import pysal
from pysal.weights import lat2W
from pysal.weights import lag_array
import numpy as np
import pylab as pl
from scipy import sparse as SP

class MomentsS():
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
                  Spatial weights instance
    u           : array
                  Residuals. nx1 array assumed to be aligned with w
 
    """
    def __init__(self,w,u):

        ut = u.T
        S = w2s(w)
        St = S.T
        StS = S * St
        D = SP.lil_matrix((w.n,w.n))
        D.setdiag(StS.diagonal())
        D = D.asformat('csr')   
        A1 = StS - D

        utSt = ut * St
        A1u = A1 * u
        Su = S * u

        g1 = np.dot(ut, A1u)
        g2 = np.dot(ut, Su)
        self.g = np.array([[g1][0][0],[g2][0][0]]) / w.n

        G11 = -2 * (np.dot(utSt, A1u)) 
        G12 = np.dot((utSt * A1), Su)
        G21 = np.dot(utSt, ((S + St) * u))
        G22 = np.dot(utSt, (S * Su))
        self.G = np.array([[G11[0][0],G12[0][0]],[G21[0][0],G22[0][0]]]) / w.n

def w2s(w):
    'Converts pysal W to scipy csr_matrix'
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
    s = SP.csr_matrix((data,indices,indptr),shape=(w.n,w.n))
    return s

if __name__ == "__main__":

    import random
    w=pysal.weights.lat2W(10,10)
    random.seed(100)
    np.random.seed(100)
    u=np.random.normal(0,1,(w.n,1))

    m=MomentsS(w,u)

