import numpy as np

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

    XT=np.transpose(X)
    xx=np.dot(XT,X)
    ixx=la.inv(xx)
    ixxx=np.dot(ixx,XT)
    b=np.dot(ixxx,y)
    yhat=np.dot(X,b)
    e=y-yhat
    n,k=X.shape
    dof=n-k
    ess=np.dot(np.transpose(e),e)
    sig2=ess/dof
    yd=y-y.mean()
    tss=np.dot(np.transpose(yd),yd)
    self.tss=tss
    self.sig2=sig2
    self.sig2ml=ess/n
    self.ess=ess
    self.dof=dof
    self.n=n
    self.k=k
    self.e=e
    self.yhat=yhat
    self.b=b
    self.ixx=ixx
    self.bvcv=sig2*ixx
    self.bse=np.sqrt(np.diag(self.bvcv))
    self.t=b/self.bse
    self.r2=1.0-ess/tss
    self.r2a=1.-(1-self.r2)*(n-1)/(n-k)
    self.lik = -0.5*(n*(np.log(2*math.pi))+n*np.log(ess/n)+(ess/(ess/n)))

    return betas

def get_residuals(x,y,betas):
    return residuals

def get_spCO(z,w,lambdaX):
    return zs

def optimizer(moments,vc=None):
    """Minimizes the moments and returns lambda"""
    return lambdaX

class Moments:
    def __init__(self,residuals,w):
        pass
    self.moments

def get_vc(residuals,w,lambdaX,moments):
    return vc


