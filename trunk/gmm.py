
from pysal.weights.spatial_lag import lag_array
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


def get_vc(w,u,l,m):
    """
    Computes the VC matrix \Psi based on \lambda and returns an array 2x2:

        psi = [psi11  psi12]
             [psi21  psi22]

    NOTE: psi12=psi21

    Parameters
    ----------
    w           : W
                  Spatial weights instance
    u           : array
                  Residuals. nx1 array assumed to be aligned with w
    l           : float
                  LambdaX. Spatial error coefficient.
    m           : Moments
                  Moments instance
                  """
    psi11=None
    psi12=None
    psi22=None

    psi=np.array([[psi11,psi12],[psi12,psi22]])

    return psi



if __name__ == "__main__":

    import pysal,random
    w=pysal.weights.lat2W(10,10)
    random.seed(100)
    np.random.seed(100)
    u=np.random.normal(0,1,(w.n,1))

    m=Moments(w,u)

