import numpy as np
import pysal
import gmm_utils as GMM
from gmm_utils import get_A1, get_spFilter
import ols as OLS
from scipy import sparse as SP
import time

class Spatial_Error_Het:
    """
    GMM method for a spatial error model with heteroskedasticity

    Based on Arraiz et al [1]_

    NOTE: w is assumed to have w.A1
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
                  (k+1)x1 array with beta estimates for intercept and x
    lamb        : float
                  Estimate of lambda
    u           : array
                  nx1 array with residuals

    References
    ----------

    .. [1] Arraiz, I., Drukker, D. M., Kelejian, H., Prucha, I. R. (2010) "A
    Spatial Cliff-Ord-Type Model with Heteroskedastic Innovations: Small and
    Large Sample Results". Journal of Regional Science, Vol. 60, No. 2, pp.
    592-614.


    """
    def __init__(self,x,y,w,cycles=1): ######Inserted i parameter here for iterations...

        #1a. OLS --> \tilde{betas}
        ols = OLS.OLS_dev(x,y)

        #1b. GMM --> \tilde{\lambda1}
        moments = GMM.moments_het(w, ols.u)
        lambda1 = GMM.optimizer_het(moments)[0][0]

        #1c. GMM --> \tilde{\lambda2}
        sigma = GMM.get_psi_sigma(w, ols.u, lambda1)
        vc1 = GMM.get_vc_het(w, sigma)
        lambda2 = GMM.optimizer_het(moments,vc1)[0][0]
        
        betas, lambda3, self.u, vc2, G = self.iterate(cycles,ols,w,lambda2)
        #Output
        #Will lambda be stacked to the betas? If not, the var-cov matrix should not contain the var of lambda.
        self.betas = np.vstack((betas,lambda3))
        self.vm = GMM.get_vm_het(G,lambda3,ols,self.u,w,vc2)

    def iterate(self,cycles,ols,w,lambda2):

        for n in range(cycles): #### Added loop.
            #2a. OLS -->\hat{betas}
            xs,ys = get_spFilter(w,lambda2,ols.x),get_spFilter(w,lambda2,ols.y)
            
            #This step assumes away heteroskedasticity, we are taking into account
            #   spatial dependence (I-lambdaW), but not heteroskedasticity
            #   GM lambda is only consistent in the absence of heteroskedasticity
            #   so do we need to do FGLS here instead of OLS?
            
            ols_i = OLS.OLS_dev(xs,ys,constant=False)

            #2b. GMM --> \hat{\lambda}
            u = ols.y - np.dot(ols.x,ols_i.betas)
            moments_i = GMM.moments_het(w, u)
            sigma_i =  GMM.get_psi_sigma(w, u, lambda2)
            vc2 = GMM.get_vc_het(w, sigma_i)
            lambda3 = GMM.optimizer_het(moments_i,vc2)[0][0]
            lambda2 = lambda3 #### 
            return ols_i.betas,lambda3,u,vc2,moments_i[0]

            #How many times do we want to iterate after 2b.? What should value of i be
            #   in loop?

# LA - note: as written, requires recomputation of spatial lag
#         for each value of lambda; since the spatial lag does
#         not change, this should be moved out

if __name__ == '__main__':

    from testing_utils import Test_Data as DAT
    data = DAT()
    y, x, w = data.y, data.x, data.w
    y = np.array([y]).T
    w.A1 = get_A1(w.sparse)
    sp = Spatial_Error_Het(x, y, w)
    print np.hstack((sp.betas,np.reshape(np.array(np.sqrt(sp.vm.diagonal())),(7,1))))

