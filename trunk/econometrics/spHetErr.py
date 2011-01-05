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

    Examples
    --------
    >>> import numpy as np
    >>> from testing_utils import Test_Data as DAT
    >>> data = DAT()
    >>> y, x, w = data.y, data.x, data.w
    >>> reg = Spatial_Error_Het(y, x, w)
    >>> print np.hstack((reg.betas,np.sqrt(reg.vm.diagonal()).reshape(7,1)))
    [[ 0.08028647  0.09883845]
     [-0.17811755  0.10271992]
     [-0.05991007  0.12766904]
     [-0.01487806  0.10826346]
     [ 0.09024685  0.09246069]
     [ 0.14459452  0.10005024]
     [ 0.00068073  0.08663456]]
    """

    def __init__(self,y,x,w,cycles=1): ######Inserted i parameter here for iterations...

        #1a. OLS --> \tilde{betas}
        ols = OLS.BaseOLS(y, x)

        #1b. GMM --> \tilde{\lambda1}
        moments = GMM.moments_het(w, ols.u)
        lambda1 = GMM.optimizer_het(moments)[0][0]

        #1c. GMM --> \tilde{\lambda2}
        sigma = GMM.get_psi_sigma(w, ols.u, lambda1)
        vc1 = GMM.get_vc_het(w, sigma)
        lambda2 = GMM.optimizer_het(moments,vc1)[0][0]
        
        betas, lambda3, self.u, vc2, G = self.iterate(cycles,ols,w,lambda2)
        #Output
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
            
            ols_i = OLS.BaseOLS(ys,xs,constant=False)

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

def _test():
    import doctest
    doctest.testmod()

if __name__ == '__main__':
    _test()
    from testing_utils import Test_Data as DAT
    data = DAT()
    y, x, w = data.y, data.x, data.w
    reg = Spatial_Error_Het(x, y, w)
    print "Dependent variable: Y"
    print "Variable  Coef.  S.E."
    print "Constant %5.4f %5.4f" % (reg.betas[0],np.sqrt(reg.vm.diagonal())[0])
    for i in range(x.shape[1]):
        print "Var_%s %5.4f %5.4f" % (i+1,reg.betas[i+1],np.sqrt(reg.vm.diagonal())[i+1])
    print "Lambda: %5.4f %5.4f" % (reg.betas[-1],np.sqrt(reg.vm.diagonal())[-1])
