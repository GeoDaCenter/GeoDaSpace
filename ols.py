import numpy as np
import numpy.linalg as la


class OLS:
    def __init__(self):
        pass


def get_betas(x,y):
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





