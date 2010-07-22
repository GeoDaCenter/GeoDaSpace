"""
Spatial Error Models module
"""
import numpy as np
import pysal
import gmm_utils as GMM
from gmm_utils import get_A1, get_spCO
import ols as OLS
import scipy.optimize as op
import numpy.linalg as la
from scipy import sparse as SP
import time


class GSLS:
    """
    Generalized Spatial Least Squares (OLS + GMM)

    To do:
        * Check optimization (initial parameters...) sigma2e does NOT match R
        * Add inference for betas
    """
    def __init__(self, x, y, w):
        w.A1 = get_A1(w.sparse)

        x = np.hstack((np.ones(y.shape),x))

        #1a. OLS --> \tilde{betas}
        ols = OLS.OLS_dev(x,y, constant=False)

        #1b. GMM --> \tilde{\lambda1}
        moments = GMM.momentsGSLS(w, ols.u)
        lambda1, sigma2e = GMM.optimizer_gsls(moments)[0]

        #2a. OLS -->\hat{betas}
        xs,ys = get_spCO(x,w,lambda1),get_spCO(y,w,lambda1)

        ols = OLS.OLS_dev(xs,ys, constant=False)

        #Output
        self.betas = ols.betas
        self.lamb = lambda1
        self.sigma2e = sigma2e
        self.u = ols.u

if __name__ == '__main__':

    #csv = pysal.open('examples/n100_stdnorm_vars6.csv','r')
    csv = pysal.open('examples/columbus.dbf','r')
    print 'csv read'
    #y = np.array([csv.by_col('varA')]).T
    y = np.array([csv.by_col('HOVAL')]).T
    print 'y read'
    #x = np.array([csv.by_col('varB'), csv.by_col('varC')]).T
    x = np.array([csv.by_col('INC'), csv.by_col('CRIME')]).T
    print 'x read'
    #w = pysal.open('examples/w_rook_n100_order1_k4.gal', 'r').read()
    w = pysal.open('examples/columbus.gal', 'r').read()
    w.transform='r' #Needed to match R

    model = GSLS(x, y, w)
    """
    print '\n'
    print model.betas
    print '\n'
    print model.lamb
    print '\n'
    print model.sigma2e
    """

    ols = OLS.OLS_dev(x,y)
    #m = GMM.momentsGSLS(w, ols.u) #MATCHES R's SPDEP

