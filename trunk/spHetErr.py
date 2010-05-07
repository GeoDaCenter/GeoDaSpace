import numpy as np
import pysal
import gmmS as GMM
import ols as OLS
import scipy.optimize as op
import numpy.linalg as la
from scipy import sparse as SP
import time

class Spatial_Error_Het:
    """GMM method for a spatial error model with heteroskedasticity"""
    def __init__(self,x,y,w,i=1): ######Inserted i parameter here for iterations...

        if not w.S:
            w.S = get_S(w)
        if not w.A1:
            w.A1 = get_A1(w.S)
        w.S = get_S(w)
        w.A1 = get_A1(w.S)

        #1a. OLS --> \tilde{betas}
        ols = OLS.OLS_dev(x,y)

        #1b. GMM --> \tilde{\lambda1}
        moments1 = GMM.Moments(ols.u,w)
        lambda1 = Optimizer(moments1).lambdaX

        #1c. GMM --> \tilde{\lambda2}
        vc1 = get_vc(ols.u,w,lambda1,moments1)
        lambda2 = Optimizer(moments1,vc1).lambdaX
        
        for n in range(i): #### Added loop.
            #2a. OLS -->\hat{betas}
            xs,ys = get_spCO(x,w,lambda2),get_spCO(y,w,lambda2)
            
            #This step assumes away heteroskedasticity, we are taking into account
            #   spatial dependence (I-lambdaW), but not heteroskedasticity
            #   GM lambda is only consistent in the absence of heteroskedasticity
            #   so do we need to do FGLS here instead of OLS?
            
            ols = OLS.OLS_dev(xs,ys)

            #2b. GMM --> \hat{\lambda}
            moments2 = GMM.Moments(ols.u,w)
            vc2 = GMM.get_vc(ols.u,w,lambda2,moments2)
            lambda3 = Optimizer(moments2,vc2).lambdaX
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

# what do we wan to pass into the Optimizer?
#          suggestion of Pedrom to do a Cholesky decomposition on the weights 
#          before computing g and G  
class Optimizer:
    """
    Finds the Lambda that minimizes the residuals from the moments conditions.
    The optimizer used is Scipy's fmin_l_bfgs_b.

    NOTE: The way it's built it uses the Cholesky Decomposition.
          We should change this.
          We can either find an optimization routine which allows
          for weights or just do the weighting prior to the optim as Nancy did.

    Parameters
    ----------

    moments_op : object?
                 should contain the Big G and the little g as attributes
    vcX        : PSI matrix for weighting
                 
    Attributes
    ----------

    lambdaX   : number?
                value of lambda at the minimum
    """

    def __init__(self, moments_op,vcX=None):
    
        
        self.moments_op=moments_op
        if vcX:
            """ Cholesky decomposition"""
            Ec = np.transpose(la.cholesky(la.inv(vcX)))
            self.moments_op.G = np.dot(Ec,moments_op.G)
            self.moments_op.g = np.dot(Ec,moments_op.g)
            
        lambdaX = op.fmin_l_bfgs_b(self.kpgm,[0.0],approx_grad=True,bounds=[(-1.0,1.0)])
        self.lambdaX = lambdaX

    def kpgm(self,lambdapar):
        """ Details:
            Gets the square residuals ready for minimization
        """
        par=np.array([float(lambdapar[0]),float(lambdapar[0])**float(2)])
        vv=np.inner(self.moments_op.G,par)
        vv2=vv-self.moments_op.g
        v2=sum(sum(vv2*vv2))
        argmin=v2
            
        return argmin

