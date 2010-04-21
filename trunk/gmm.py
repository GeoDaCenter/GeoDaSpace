
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
    
    def __init__(self,w,u,symmetric=False):

        ul=lag_array(w,u)
        ull=lag_array(w,ul)
        ultu=np.dot(ul.transpose(),u)
        ultul=np.dot(ul.transpose(),ul)
        ultull=np.dot(ul.transpose(),ull)
        ulltu=np.dot(ull.transpose(),u)
        ulltul=np.dot(ull.transpose(),ul)
        ulltull=np.dot(ull.transpose(),ull)
        ni=(1./w.n)

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


        g1=ni*(ultul-sum([d[i]*(u[i]**2) for i in range(w.n)]))
        g2=ni*ultu
        self.g=np.array([[g1][0][0],[g2][0][0]])

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
    """Computes the VC matrix \Psi based on \lambda and returns an array 2x2:

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
    psi11=
    psi12=
    psi22=

    psi=np.array([[psi11,psi12],[psi12,psi22]])

    return psi



if __name__ == "__main__":

    import pysal,random
    w=pysal.weights.lat2W(10,10)
    random.seed(100)
    np.random.seed(100)
    u=np.random.normal(0,1,(w.n,1))

    m=Moments(w,u)

