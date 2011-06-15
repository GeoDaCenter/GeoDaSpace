'''
Empirical copula
'''

import numpy as np
import scipy.stats as ss
import matplotlib.pyplot as plt
import random, time
import scipy as sp
import pysal as ps
from scatter_plot import scatter_pts

def pec(v, l=10):
    n, p = v.shape
    step = 1./l
    u = np.arange(0, 1.+step, step)
    C = np.zeros((l+1, l+1))
    for i in range(l+1):
        for j in range(l+1):
            C[i, j] = pec_single(v, np.array([u[i], u[j]]))
    return C, u

def pec_single(v, u):
    n, p = v.shape
    res = 0.
    for i in range(n):
        ind = 1.
        for j in range(p):
            F_j_X_ij = ecdf(v[i, j], v[:, j], n) * n / (n+1)
            ind *= I(F_j_X_ij, u[j])
        res += ind
    return res / n

def dec(v, l=10):
    C, u = pec(v, l=l)
    c = np.zeros(C.shape)
    lu = len(u)
    for i in range(1, lu):
        for j in range(1, lu):
            c[i, j] = C[i, j] - C[i-1, j] - C[i, j-1] + C[i-1, j-1]
    return c, u

def ecdf(u, U, n):
    return len([i for i in U if i <= u]) / float(n)

def ecdf_r(U, n):
    'Faster ecdf for univariate'
    return ss.rankdata(U) / n

def I(x, th):
    'Indicator function'
    if x <= th:
        return 1
    else:
        return 0

if __name__ == '__main__':
    #random.seed(123)
    v = np.array([[-0.56047565, -0.23017749,  1.55870831,  0.07050839,
        0.12928774,  1.71506499, 0.46091621, -1.26506123, -0.68685285,
        -0.44566197]]).T
    '''
    times = []
    times_r = []
    for i in range(10):
        v = sp.random.random((100, 1)) + 10
        t0 = time.time()
        e = [ecdf(i, v, len(v)) for i in v]
        t1 = time.time()
        er = ecdf_r(v, len(v))
        t2 = time.time()
        for i,j in zip(e, er):
            if i != j:
                print i, '\t', j
        times.append(t1 - t0)
        times_r.append(t2 - t1)
    '''
    #print np.mean(times), '\t', np.mean(times_r)
    #v = sp.random.random((100, 2)) + 1
    #v = np.hstack((v, v**3))
    w = ps.lat2W(10, 10)
    y, wy = scatter_pts(w, rho=0.15)
    v = np.hstack(tuple([np.array([i]).T for i in [y, wy]]))
    #ec, u = pec(v, l=10)
    #print ec
    ec, u = dec(v, l=5)
    print ec
    #con = plt.contourf(u, u, ec)
    im = plt.imshow(ec[1:, 1:], origin='lower')
    plt.colorbar()
    plt.show()

