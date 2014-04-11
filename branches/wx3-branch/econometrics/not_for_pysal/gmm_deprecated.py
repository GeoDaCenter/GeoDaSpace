"""
Tools for different GMM procedure estimations using full numpy arrays and
PySAL original weights structures

NOTE: its use is deprecated
"""

#from pysal.weights.spatial_lag import lag_array
from scipy import sparse as SP
import numpy as np

class Moments():
    """
    Moments class to compute all six components of the system of equations for
    a spatial error model with heteroskedasticity estimated by GMM

    It implements eq. 30 in olsGMMwriteOut.pdf

    [g1] + [G11 G12] *  [\lambda]    = [0]
    [g2]   [G21 G22]    [\lambda^2]    [0]

    NOTE: 'residuals' has been renamed 'u' to fit paper notation

    Parameters
    ----------
    w           : W
                  Spatial weights instance
    u           : array
                  Residuals. nx1 array assumed to be aligned with w
                  """
    
    # LA - d should be computed outside of this
    #         whereas u can change in an iterative procedure,
    #         the value of d does not and we don't want to recompute this
    #         each time
    
    def __init__(self,w,u,symmetric=False):

        ul=lag_array(w,u)
        ult=ul.transpose()
        ull=lag_array(w,ul)
        ullt=ull.transpose()

        # LA - use np.inner(ul,u)? why transpose a vector?
        ultu=np.dot(ult,u)
        ultul=np.dot(ult,ul)
        ultull=np.dot(ult,ull)
        ulltu=np.dot(ullt,u)
        ulltul=np.dot(ullt,ul)
        ulltull=np.dot(ullt,ull)
        # LA - is unnecessary extra step, just divide by w.n where needed
        ni=(1./w.n)

        # move outside of this class
        d=np.zeros((w.n,1))
        if symmetric:
            for i in range(w.n):
                d[i]=sum(np.power(w.weights[w.id_order[i]]))
        else:
            for i in range(w.n):
                di=0
                for j in w.neighbor_offsets[i]:
                    w_ji=w.weights[w.id_order[j]][w.neighbor_offsets[j].index(i)]
                    di+=w_ji**2
                d[i]=di

        # no need to premultiply by ni, see below
        g1=ni*(ultul-sum([d[i]*(u[i]**2) for i in range(w.n)]))
        g2=ni*ultu
        self.g=np.array([[g1][0][0],[g2][0][0]])  # divide by n after this

        G11=2*ni*(sum([d[i]*u[i]*ul[i] for i in range(w.n)])-ultull) 
        G12=ni*(ulltull-sum([d[i]*(ul[i]**2) for i in range(w.n)]))
        G21=ni*(ultul+ulltu)
        G22=ni*ulltul
        self.G=np.array([[G11[0][0],G12[0][0]],[G21[0][0],G22[0][0]]])

        self.ul=ul
        self.ull=ull
        self.ultu=ultu
        self.ultul=ultul
        self.ultull=ultull
        self.ulltu=ulltu
        self.ulltul=ulltul
        self.ulltull=ulltull
        self.d=d

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

def get_vcF(w, u, l):
    """
    Computes the VC matrix Psi based on lambda using full numpy arrays:

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

    import pysal,random
    w=pysal.weights.lat2W(10,10)
    random.seed(100)
    np.random.seed(100)
    u=np.random.normal(0,1,(w.n,1))

    m=Moments(w,u)

