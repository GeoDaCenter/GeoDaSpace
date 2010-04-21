import numpy as np
import pysal
import gmm as GMM
import ols as OLS

class Spatial_Error_Het:
    """GMM method for a spatial error model with heteroskedasticity"""
    def __init__(self,x,y,w):

        #1a. OLS --> \tilde{betas}
        betas1=OLS.get_betas(x,y)

        #1b. GMM --> \tilde{\lambda1}
        residuals1=OLS.get_residuals(x,y,betas1)
        moments1=GMM.Moments(residuals1,w)
        lambda1=optimizer(moments1.moments)

        #1c. GMM --> \tilde{\lambda2}
        vc1=get_vc(residuals1,w,lambda1,moments1)
        lambda2=optimizer(moments1,vc1)

        #2a. OLS -->\hat{betas}
        xs,ys=get_spCO(x,w,lambda2),get_spCO(y,w,lambda2)
        betas2=OLS.get_betas(xs,ys)

        #2b. GMM --> \hat{\lambda}
        residuals2=OLS.get_residuals(xs,ys,betas2)
        moments2=GMM.Moments(residuals2,w)
        vc2=GMM.get_vc(residuals2,w,lambda2,moments2)
        lambda3=optimizer(moments2.moments,vc2)

        #Output
        self.betas=betas2
        self.lamb=lambda3
        self.residuals=residuals2

        
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



