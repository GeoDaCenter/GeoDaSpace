'''
Playing with multiprocessing to offload trace computation
'''

from multiprocessing import Process
import numpy as np
import pysal as ps

def trWpW(w):
    tr = w.sparse.T * w.sparse
    return np.sum(tr.diagonal())

if __name__ == '__main__':

    w = ps.lat2W(1750, 1750)
    print 'N: %i'%w.n
    tr0 = trWpW(w)
    print 'Trace computed in current thread'
    p = Process(target=trWpW, args=(w,))
    p.start()
    p.join()
    print 'Trace computed in offloaded process'
