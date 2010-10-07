"""
Example of a simulation split into several cores

In this case, we are obtaining the average value for several draws of a random
distribution

ToDo's:

    * line 55: pool.map Is there an efficient number of elements for the map?
    * Code and compare pool.map method with 'manual' multiprocessing
"""

import multiprocessing as mp
import numpy as np
import time

def sim_draw(n):
    x = np.random.randn(n)
    return np.mean(x)

def worker_task(cycles):
    """
    Example of function to be run by one core

    Parameters
    ----------

    cycles  : int
              Number of cycles to repeat the task
    """
    values_core = np.zeros((cycles, 1))
    for i in xrange(cycles):
        values_core[i] = sim_draw(10)
    return values_core

def output_paster(values_cores):
    """
    Merges output from different cores in one object

    Parameters
    ----------

    values_cores    : list
                      List of arrays out of different cores
    """
    return np.vstack([i for i in values_cores])

if __name__ == '__main__':

    welcome = """
    Computation time to perform 100000 draws of 10 random values and keep the
    average
    """
    print welcome
    cores = mp.cpu_count()
    cycles_pc = 100000
    pool = mp.Pool(cores)
    t0 = time.time()
    parts = pool.map(worker_task, [cycles_pc/cores] * cores)
    means = output_paster(parts)
    t1 = time.time()
    print '\nRun on multi-core: %.4f seconds'%(t1-t0)
    print 'Average value: ', np.mean(means)
    single = worker_task((cycles_pc/cores) * cores)
    t2 = time.time()
    print '\nRun on single-core: %.4f seconds'%(t2-t1)
    print 'Average value: ', np.mean(single)
    print '\nNumber of iterations actually performed:', (cycles_pc/cores) * cores


