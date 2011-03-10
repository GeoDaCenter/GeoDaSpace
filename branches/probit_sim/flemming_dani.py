'''
GMM estimation of Spatial Error models for discrete dependent variable as in
Flemming
'''

import numpy as np
import pysal as ps
from scipy.stats import norm
from scipy import sparse as sp
import scipy.optimize as op

class ProbitGMMerr:
    def __init__(self, y, x, w):
        self.y = y
        self.x = x
        self.w = w
        self.n, self.k = x.shape

        self.betas = op.fmin_l_bfgs_b(self._s,[0.0]*self.k,args=[y, x],approx_grad=True,bounds=[(None, None), (None, None)])
        

    def _s(self, betas, y, x):
        '''
        S as in p.164 for M = I
        '''
        xb = np.dot(x, betas)
        m = np.dot(x.T * self._a(xb), (y - norm.cdf(xb))) / self.n
        return np.dot(m.T, m)

    def _a(self, xb):
        '''
        Builds A as in eq. 7.33 in csr sparse format
        '''
        d = norm.pdf(xb) / (norm.cdf(xb) * norm.sf(xb))
        s = sp.lil_matrix((len(d), len(d)))
        s.setdiag(d)
        return s.tocsr()

if __name__ == '__main__':
    w = ps.lat2W(10, 10)
    r = np.random.random((100, 1))
    c = np.ones((100, 1))
    x = np.hstack((c, r))
    e = np.random.random((100, 1))
    betas = np.array([[1.] * x.shape[1]]).T
    y_lat = np.dot(x, betas) + e
    y = map(round, np.random.random((y_lat.shape)))
    
    p = ProbitGMMerr(y, x, w)


