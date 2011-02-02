'''
Script to test scalability of inverse distance weights in PySAL
'''

import pysal, time
import numpy as np
import multiprocessing as mp
from matplotlib import pyplot as pl

def randomInvDistW(n, band=5, p=2., binary=False, alpha=-1.0):
    '''
    Builds a W object from random points based on knn
    ...

    Arguments
    ---------
    n       : int
              Number of observations for the W object
    band    : int
              Distance band
    p       : float
              Minkowski p-norm distance metric parameter
    binary  : boolean
              If true w_{ij}=1 if d_{i,j}<=threshold, otherwise w_{i,j}=0
              If false wij=dij^{alpha}
    alpha   : float
              distance decay parameter for weight (default -1.0)
              if alpha is positive the weights will not decline with
              distance. If binary is True, alpha is ignored

    Returns
    -------
    nt      : tuple
              Size and time respectively elapsed packed in a tuple

    '''
    print 'Starting %i observations'%n
    t = None
    try:
        t0 = time.time()
        xy = np.random.random(n * 2)
        xy = xy.reshape((n, 2))
        w = pysal.DistanceBand(xy, 5, p, alpha=alpha, binary=binary)
        t1 = time.time()
        t = t1-t0
        res = (n, t)
    except:
        res = (n, 0.)
    print '\t\tSize %i run correctly in %.5f seconds'%(n, t)
    return res

def glue(parts):
    """
    Merges output from different cores in one object

    Parameters
    ----------

    values_cores    : list
                      List of arrays out of different cores
    """
    l = []
    for i in parts:
        l.extend(i)
    return l

def paral_sim(sizes):
    '''
    Runs the simulation on multiple cores
    ...

    Arguments
    ---------
    sizes       : list
                  Sequence of sizes to test distances
    '''
    cores = mp.cpu_count()
    pool = mp.Pool(cores)
    return pool.map(randomInvDistW, sizes)

sizes = [5000, 6000, 7000, 8000, 9000, 10000, 11000, 12000, 13000, 14000, \
        15000, 16000, 17000, 18000, 19000, 20000, 21000, 22000, 23000, \
        24000, 25000]

#sizes = [i/100 for i in sizes][:5]

stats = paral_sim(sizes)
xy = np.array(stats)

fig = pl.figure()
p = fig.add_subplot(111)
p.plot(xy[0], xy[1])
p.set_xlabel('N')
p.set_ylabel('Seconds')
pl.savefig('invDist.png')
#pl.show()

