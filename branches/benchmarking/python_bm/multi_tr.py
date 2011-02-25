'''
Playing with multiprocessing to offload trace computation
'''

from multiprocessing import Process
import multiprocessing as mp
import numpy as np
import pysal as ps
import time

def trWpW(w):
    tr = w.T * w
    return np.sum(tr.diagonal())

if __name__ == '__main__':

    s = 3000
    x = np.random.random((s**2, 100))
    xx = np.random.random((s**2, 100))
    y = np.random.random((s**2, 1))
    w = ps.weights.lat2SW(s, s)
    print 'N: %i'%s**2
    tr0 = trWpW(w)
    print 'Trace computed in current thread'
    '''
    w2 = ps.weights.lat2SW(2000, 2000, criterion='rook')
    print 'Created new w'
    tr1 = trWpW(w2)
    print 'Trace computed in current thread'
    p = Process(target=trWpW, args=(w2,))
    p.start()
    p.join()
    cores = mp.cpu_count()
    pool = mp.Pool(cores)
    trs = pool.map(trWpW, [w2])
    '''
    print 'Trace computed in offloaded process'
