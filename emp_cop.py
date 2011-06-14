'''
Empirical copula
'''

import numpy as np
import scipy.stats as ss
import matplotlib.pyplot as plt
import random, time
import scipy as sp

def pec(v):
    n = U.shape[0]
    U = ss.rankdata(U)
    C = np.zeros((n, 1))
    for i in range(n):
        ind = 1.
        for j in range(n):
        ind *= 
    return C

def ecdf(u, U, n):
    return len([i for i in U if i <= u]) / float(n)

def ecdf_r(U, n):
    'Faster ecdf for univariate'
    return ss.rankdata(U) / n

if __name__ == '__main__':
    #random.seed(123)
    v = np.array([[-0.56047565, -0.23017749,  1.55870831,  0.07050839,
        0.12928774,  1.71506499, 0.46091621, -1.26506123, -0.68685285,
        -0.44566197]]).T
    times = []
    times_r = []
    for i in range(1000):
        v = sp.random.random((100, 1))
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
    print np.mean(times), '\t', np.mean(times_r)

