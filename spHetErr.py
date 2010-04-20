import numpy as np
import numpy.linalg as la
import pysal

class SpHetErr:
    """GMM method for a spatial error model with heteroskedasticity"""
    def __init__(self,x,y,w):

        #1a. OLS --> \tilde{betas}
        betas1=ols(x,y)

        #1b. GMM --> \tilde{\lambda1}
        residuals1=get_residuals(x,y,betas1)
        moments1=Moments(residuals1,w)
        lambda1=optimizer(moments1.moments)

        #1c. GMM --> \tilde{\lambda2}
        vc1=get_vc(residuals1,w,lambda1,moments1)
        lambda2=optimizer(moments1,vc1)

        #2a. OLS -->\hat{betas}
        xs,ys=get_spCO(x,w,lambda2),get_spCO(y,w,lambda2)
        betas2=ols(xs,ys)

        #2b. GMM --> \hat{\lambda}
        residuals2=get_residuals(xs,ys,betas2)
        moments2=Moments(residuals2,w)
        vc2=get_vc(residuals2,w,lambda2,moments2)
        lambda3=optimizer(moments2.moments,vc2)

        #Output
        self.betas=betas2
        self.lamb=lambda3
        self.residuals=residuals2

        
 
def ols(x,y):
    xt=np.transpose(x)
    xx=np.dot(xt,x)
    ixx=la.inv(xx)
    ixxx=np.dot(ixx,xt)
    betas=np.dot(ixxx,y)
    return betas

def get_residuals(x,y,betas):
	predy=np.dot(x,betas)
	residuals=y-predy
    return residuals

def get_spCO(z,w,lambdaX):
	lagz=pysal.weights.spatial_lag.lag_array(w,z)
	zs=z-lambdaX*lagz
    return zs

def optimizer(moments,vc=None):
    """Minimizes the moments and returns lambda"""
    return lambdaX

class Moments:
    def __init__(self,w,residuals):
        pass
    self.moments

def get_vc(residuals,w,lambdaX,moments):
    return vc


