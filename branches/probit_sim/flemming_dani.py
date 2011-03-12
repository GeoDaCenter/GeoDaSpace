'''
GMM estimation of Spatial Error models for discrete dependent variable as in
Flemming
'''

import numpy as np
import pysal as ps
from scipy.stats import norm
from scipy import sparse as sp
import scipy.optimize as op
from econometrics.probit import probit

class ProbitGMMerr:
    def __init__(self, y, x, w, uused='gen'):
        self.y = y
        self.x = x
        w.transform = 'r'
        self.w = w
        self.n, self.k = x.shape

        self.betas = probit(y, x, constant=False).betas
        print 'Betas Probit'
        print self.betas
        y = y[:, 0]
        self.betas = op.fmin_l_bfgs_b(self._s,[0.0]*self.k,args=[y, x],approx_grad=True,bounds=[(None, None), (None, None)])[0]
        print 'Betas GMM'
        print self.betas

        self.lat_y = np.dot(x, self.betas)
        predy = norm.cdf(self.lat_y)
        u_naive = y - predy
        u = u_naive

        if uused == 'gen':
            phiy = norm.pdf(self.lat_y)
            Phi_prod = predy * (1 - predy)
            u_gen = phiy * (u_naive / Phi_prod)
            u = u_gen

        u = u.reshape((w.n, 1))
        # KP
        moments = self._momentsGMSWLS(w, u)
        self.lamb = self._optimizer_gmswls(moments)[0][0]


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

    def _momentsGMSWLS(self, w, u):

        u2 = np.dot(u.T, u)
        wu = w.sparse * u
        uwu = np.dot(u.T, wu)
        wu2 = np.dot(wu.T, wu)
        wwu = w.sparse * wu
        uwwu = np.dot(u.T, wwu)
        wwu2 = np.dot(wwu.T, wwu)
        wuwwu = np.dot(wu.T, wwu)
        wtw = w.sparse.T * w.sparse
        trWtW = np.sum(wtw.diagonal())

        g = np.array([[u2[0][0], wu2[0][0], uwu[0][0]]]).T / w.n

        G = np.array([[2 * uwu[0][0], -wu2[0][0], w.n], [2 * wuwwu[0][0], -wwu2[0][0], trWtW], [uwwu[0][0] + wu2[0][0], -wuwwu[0][0], 0.]]) / w.n

        return [G, g]

    def _optimizer_gmswls(self, moments):
        """
        Optimization of moments
        ...

        Parameters
        ----------

        moments     : _momentsGMSWLS
                      Instance of _momentsGMSWLS with G and g
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
           
        lambdaX = op.fmin_l_bfgs_b(self._gm_gmswls,[0.0, 0.0],args=[moments],approx_grad=True,bounds=[(-1.0,1.0), (0, None)])
        return lambdaX

    def _gm_gmswls(self, lambdapar, moments):
        """
        Preparation of moments for minimization in a GMSWLS framework
        """
        par=np.array([[float(lambdapar[0]),float(lambdapar[0])**2., lambdapar[1]]]).T
        vv = np.dot(moments[0], par)
        vv = vv - moments[1]
        return sum(vv**2)

if __name__ == '__main__':

    import sys
    sys.path.append('/Users/pedroamaral/Documents/Academico/GeodaCenter/python/SVN/spreg/trunk/')
    from econometrics import power_expansion as PE
    np.random.seed(10)
    w = ps.lat2W(30, 30)
    w.transform='r'
    r = np.random.normal( -1, 1, (900, 1))
    c = np.ones((900, 1))
    x = np.hstack((c, r))
    betas = np.array([[1.] * x.shape[1]]).T

    e = np.random.normal(0,1,(w.n,1)) #Build residuals vector
    ys = np.dot(x, betas) + PE.power_expansion(w, e, 0.5) #Build y_{star}    
    y = np.zeros((w.n, 1),float) #Binary y
    for yi in range(len(y)):
        if ys[yi]>0:
            y[yi] = 1    
    
    p = ProbitGMMerr(y, x, w, uused='naive')
    print 'Lambda: %f'%p.lamb


