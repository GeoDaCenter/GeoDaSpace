'''
Empirical copula
'''

import numpy as np
import scipy.stats as ss
import matplotlib.pyplot as plt
import random, time
import scipy as sp
import pysal as ps
import multiprocessing as mp
from scatter_plot import scatter_pts

def pec_multi(v, l=10):
    '''
    Todo: find an elegant way of not computing first row of densities when we
        know it's 0
    '''
    n, p = v.shape
    correc = n / (n+1.)
    u = np.linspace(0, 1, l+1)
    ecdfs = np.array([ecdf_r(v[:, i], n) for i in range(p)]).T
    F_j_X_ijS = ecdfs * correc
    cores = mp.cpu_count()
    pool = mp.Pool(cores)
    seps = map(int, np.linspace(0, l+1, cores+1))
    seps = [(seps[i], seps[i+1]) for i in range(len(seps)-1)]
    pars_tuples = [(i, v, u, F_j_X_ijS, l) for i in seps]
    blocks = pool.map(block_C, pars_tuples)
    return np.vstack(tuple(blocks)), u

def block_C(pars_tuple):
    ran, v, u, F_j_X_ijS, l = pars_tuple
    C = np.zeros((ran[1]-ran[0], l+1))
    for i in range(*ran):
        for j in range(1, l+1):
            C[i-ran[0], j] = pec_single(v, np.array([u[i], u[j]]), F_j_X_ijS)
    return C

def pec(v, l=10):
    n, p = v.shape
    correc = n / (n+1.)
    u = np.linspace(0, 1, l+1)
    C = np.zeros((l+1, l+1))
    ecdfs = np.array([ecdf_r(v[:, i], n) for i in range(p)]).T
    F_j_X_ijS = ecdfs * correc
    for i in range(1, l+1):
        for j in range(1, l+1):
            C[i, j] = pec_single(v, np.array([u[i], u[j]]), F_j_X_ijS)
    return C, u

def pec_slow(v, l=10):
    n, p = v.shape
    u = np.linspace(0, 1, l+1)
    C = np.zeros((l+1, l+1))
    for i in range(l+1):
        for j in range(l+1):
            C[i, j] = pec_single_slow(v, np.array([u[i], u[j]]))
    return C, u

def pec_single(v, u, F_j_X_ijS):
    n, p = v.shape
    res = 0.
    for i in range(n):
        ind = 1.
        for j in range(p):
            if F_j_X_ijS[i, j] > u[j]:
                ind = 0.
                break
        res += ind
    return res / n

def pec_single_slow(v, u):
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
    return c, u, C

def dec_multi(v, l=10):
    C, u = pec_multi(v, l=l)
    cores = mp.cpu_count()
    pool = mp.Pool(cores)
    seps = map(int, np.linspace(0, l+1, cores+1))
    seps = [(seps[i], seps[i+1]) for i in range(len(seps)-1)]
    pars_tuples = [(i, C) for i in seps]
    blocks = pool.map(block_c, pars_tuples)
    c = np.vstack(tuple(blocks))
    c[0, :] = np.zeros((1, l+1))
    return c, u, C

def block_c(pars_tuple):
    ran, C = pars_tuple
    c = np.zeros((ran[1]-ran[0], l+1))
    for i in range(*ran):
        for j in range(1, l+1):
            c[i-ran[0], j] = C[i, j] - C[i-1, j] - C[i, j-1] + C[i-1, j-1]
    return c

def dec_slow(v, l=10):
    C, u = pec_slow(v, l=l)
    c = np.zeros(C.shape)
    lu = len(u)
    for i in range(1, lu):
        for j in range(1, lu):
            c[i, j] = C[i, j] - C[i-1, j] - C[i, j-1] + C[i-1, j-1]
    return c, u, C

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
    #w = ps.lat2W(10, 10)
    times = []
    times_r = []
    l = 50
    for i in range(1):
        v = sp.random.random((1000, 2)) * 10
        t0 = time.time()
        #e = [ecdf(i, v, len(v)) for i in v]
        print 'Running multi'
        c_orig = dec_multi(v, l=l)
        #c_orig = dec_slow(v)
        t1 = time.time()
        print 'Running single'
        #er = ecdf_r(v, len(v))
        c_opt = dec(v, l=l)
        t2 = time.time()
        np.testing.assert_array_equal(c_orig[0], c_opt[0])
        times.append(t1 - t0)
        times_r.append(t2 - t1)
    print np.mean(times), '\t', np.mean(times_r)
    print np.round(np.mean(times) / np.mean(times_r), decimals=2), 'times faster'
    '''
    v0 = (sp.random.random((100, 1)))
    v0.sort(axis=0)
    vv = (sp.random.random((100, 1)))
    vv[25:75] = v0[25:75]
    v = np.hstack((v0, vv))
    #y, wy = scatter_pts(w, rho=0.75)
    print 'Points gotten'
    #m = ps.Moran(y, w)
    #print m.I, m.p_sim
    #v = np.hstack(tuple([np.array([i]).T for i in [y, wy]]))
    #ec, u = pec(v, l=10)
    #print ec
    l = 15
    ec, u, C = dec(v, l=l)
    print 'Copula run'
    #print ec
    #con = plt.contourf(u, u, ec)
    #im = plt.imshow(ec[1:, 1:], origin='lower')
    ct = plt.contourf(u[1:], u[1:], ec[1:, 1:]) 
    plt.colorbar()
    plt.show()
    '''

