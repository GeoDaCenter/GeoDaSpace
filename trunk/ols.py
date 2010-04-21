import numpy as np
import numpy.linalg as la


class OLS:
    def __init__(self):
        pass

# LA - should probably be get_olsbetas(x,y)
def get_betas(x,y):
    xt=np.transpose(x)
    xx=np.dot(xt,x)
    ixx=la.inv(xx)
    # LA - preferred is to compute xy = np.dot(xt,y) first
    # LA - next betas = np.dot(ixx,xy)
    ixxx=np.dot(ixx,xt)
    betas=np.dot(ixxx,y)
    # LA - in addition to betas, return ixx, used to compute var(betas)
    return betas

# LA - note: residuals are generic for any linear model,
def get_residuals(x,y,betas):
	predy=np.dot(x,betas)
	residuals=y-predy
	# LA - in addition to residuals, return predicted values
    return residuals





