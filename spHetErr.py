import numpy as np
import pysal
import gmmS as GMM
import ols as OLS
import scipy.optimize as op
import numpy.linalg as la
from scipy import sparse as SP
import time

class Spatial_Error_Het:
    """
    GMM method for a spatial error model with heteroskedasticity

    NOTE: w is assumed to have w.S and w.A1
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

    """
    def __init__(self,x,y,w,cycles=1): ######Inserted i parameter here for iterations...

        #1a. OLS --> \tilde{betas}
        ols = OLS.OLS_dev(x,y)

        #1b. GMM --> \tilde{\lambda1}
        moments = GMM.Moments(w, ols.u)
        lambda1 = optimizer(moments)[0][0]

        #1c. GMM --> \tilde{\lambda2}
        vc1 = GMM.get_vc(w, ols.u, lambda1)
        lambda2 = optimizer(moments,vc1)[0][0]
        
        for n in range(cycles): #### Added loop.
            #2a. OLS -->\hat{betas}
            xs,ys = get_spCO(x,w,lambda2),get_spCO(y,w,lambda2)
            
            #This step assumes away heteroskedasticity, we are taking into account
            #   spatial dependence (I-lambdaW), but not heteroskedasticity
            #   GM lambda is only consistent in the absence of heteroskedasticity
            #   so do we need to do FGLS here instead of OLS?
            
            ols = OLS.OLS_dev(xs,ys)

            #2b. GMM --> \hat{\lambda}
            moments = GMM.Moments(w, ols.u)
            vc2 = GMM.get_vc(w, ols.u, lambda2)
            lambda3 = optimizer(moments,vc2)[0][0]
            lambda2 = lambda3 #### 

            #How many times do we want to iterate after 2b.? What should value of i be
            #   in loop?
        
        #Output
        self.betas = ols.betas
        self.lamb = lambda3
        self.u = ols.u

        
# LA - note: as written, requires recomputation of spatial lag
#         for each value of lambda; since the spatial lag does
#         not change, this should be moved out
def get_spCO(z,w,lambdaX):
    """
    Spatial Cochrane-Orcut Transf

    ...

    Parameters
    ----------

    Returns
    -------

    """
    return z - lambdaX * (w.S * z)

def get_S(w):
    """
    Converts pysal W to scipy csr_matrix
    ...

    Parameters
    ----------

    w               : W
                      Spatial weights instance

    Returns
    -------

    Implicit        : csr_matrix
                      PySAL W object converted into Scipy sparse matrix
                
    """
    data = []
    indptr = [0]
    indices = []
    for ob in w.id_order:
        data.extend(w.weights[ob])
        indptr.append(indptr[-1] + len(w.weights[ob]))
        indices.extend(w.neighbors[ob])
    data = np.array(data)
    indices = np.array(indices)
    indptr = np.array(indptr)
    return SP.csr_matrix((data,indices,indptr),shape=(w.n,w.n))

def get_A1(S):
    """
    Builds A1 as in Arraiz et al.

    .. math::

        A_1 = W' W - diag(w'_{.i} w_{.i})

    ...

    Parameters
    ----------

    S               : csr_matrix
                      PySAL W object converted into Scipy sparse matrix

    Returns
    -------

    Implicit        : csr_matrix
                      A1 matrix in scipy sparse format
                
    """
    StS = S.T * S
    d = SP.spdiags([StS.diagonal()], [0], S.get_shape()[0], S.get_shape()[1])
    d = d.asformat('csr')
    return StS - d

def optimizer(moments, vcX=np.array([0])):
    """
    Optimization of moments
    ...

    Parameters
    ----------

    moments     : Moments
                  Instance of gmmS.Moments with G and g
    vcX         : array
                  Optional. 2x2 array with the Variance-Covariance matrix to be used as
                  weights in the optimization (applies Cholesky
                  decomposition). Set empty by default.

    Returns
    -------
    x, f, d     : tuple
                  x -- position of the minimum
                  f -- value of func at the minimum
                  d -- dictionary of information from routine
                        d['warnflag'] is
                            0 if converged
                            1 if too many function evaluations
                            2 if stopped for another reason, given in d['task']
                        d['grad'] is the gradient at the minimum (should be 0 ish)
                        d['funcalls'] is the number of function calls made
    """
    if vcX.any():
        Ec = np.transpose(la.cholesky(la.inv(vcX)))
        moments.G = np.dot(Ec,moments.G)
        moments.g = np.dot(Ec,moments.g)
        
    lambdaX = op.fmin_l_bfgs_b(kpgm,[0.0],args=[moments],approx_grad=True,bounds=[(-1.0,1.0)])
    return lambdaX

def kpgm(lambdapar,moments):
    """ 
    Preparation of moments for minimization
    ...

    Parameters
    ----------

    lambdapar       : float
                      Spatial autoregressive parameter
    moments         : Moments
                      Instance of gmmS.Moments with G and g

    Returns
    -------

    Implicit        : float
                      sum of square residuals (e) of the equation system 
                      moments.g - moments.G * lambdapar = e
    """
    par=np.array([float(lambdapar[0]),float(lambdapar[0])**float(2)])
    vv=np.inner(moments.G,par)
    vv2=vv-moments.g
    return sum(sum(vv2*vv2))

if __name__ == '__main__':

    from testing_utils import Test_Data as DAT
    data = DAT()
    y, x, w = data.y, data.x, data.w
    y = np.array([y]).T
    w.S = get_S(w)
    w.A1 = get_A1(w.S)
    sp = Spatial_Error_Het(x, y, w)

