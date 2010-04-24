import numpy as np
import pysal
import gmm as GMM
import ols as OLS

class Spatial_Error_Het:
    """GMM method for a spatial error model with heteroskedasticity"""
    def __init__(self,x,y,w):

        #1a. OLS --> \tilde{betas}
        ols = OLS.OLS_dev(x,y)

        #1b. GMM --> \tilde{\lambda1}
        moments1 = GMM.Moments(ols.u,w)
        lambda1 = optimizer(moments1.moments)

        #1c. GMM --> \tilde{\lambda2}
        vc1 = get_vc(ols.u,w,lambda1,moments1)
        lambda2 = optimizer(moments1,vc1)

        #2a. OLS -->\hat{betas}
        xs,ys = get_spCO(x,w,lambda2),get_spCO(y,w,lambda2)
        ols = OLS.OLS_dev(xs,ys)

        #2b. GMM --> \hat{\lambda}
        moments2 = GMM.Moments(ols.u,w)
        vc2 = GMM.get_vc(ols.u,w,lambda2,moments2)
        lambda3 = optimizer(moments2.moments,vc2)

        #Output
        self.betas = ols.betas
        self.lamb = lambda3
        self.u = ols.u

        
# LA - note: as written, requires recomputation of spatial lag
#         for each value of lambda; since the spatial lag does
#         not change, this should be moved out
def get_spCO(z,w,lambdaX):
	lagz=pysal.weights.spatial_lag.lag_array(w,z)
	zs=z-lambdaX*lagz
    return zs

def optimizer(moments,vc=None):
    """Minimizes the moments and returns lambda"""
    return lambdaX



