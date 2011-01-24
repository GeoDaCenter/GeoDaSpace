"""
Script to create large datasets and benchmark times
"""

import pysal as ps
import time
import numpy as np
from pysal.spreg.ols import OLS
from pysal.spreg.ols import BaseOLS
from pysal.spreg.diagnostics_sp import LMtests, MoranRes
from econometrics.testing_utils import Test_Data as Data

print '\n\t\t\t### Large simulated dataset benchmarking ###\n'

def test_large_all(s, k, log):
    """
    Run and time OLS and various diagnostics

    Arguments
    ---------
    s       : int
              Side of the lattice from which to build the weights on
              n = s**2
    k       : int
              N of variables to include in regression
    log     : string
              Path of log file to append results to

    Returns
    -------
              Updates on-the-fly the log file with new timing results
    """
    log = open(log, 'a')
    n = 'n: %i\n'%s**2
    log.write(n)
    print n
    vars = 'k: %i\n'%k
    log.write(vars)
    print vars

    ti = time.time()
    t0 = time.time()
    data = Data(s**2, k, 'large', '../../../trunk/econometrics/', omit_w=True)
    t1 = time.time()
    tf = t1 - t0
    creDa = 'Create data:\t\t\t%.5f seconds\n'%tf
    log.write(creDa)
    print creDa

    t0 = time.time()
    w = ps.lat2W(s, s, rook=False)
    t1 = time.time()
    tf = t1 - t0
    creWe = 'Created Weights:\t\t%.5f seconds\n'%tf
    log.write(creWe)
    print creWe

    t0 = time.time()
    ols = OLS(data.y, data.x)
    t1 = time.time()
    tf = t1 - t0
    runOls = 'Regression:\t\t\t%.5f seconds\n'%tf
    log.write(runOls)
    print runOls

    t0 = time.time()
    lms = LMtests(ols, w)
    t1 = time.time()
    tf = t1 - t0
    runLm = 'LM diagnostics:\t\t\t%.5f seconds\n'%tf
    log.write(runLm)
    print runLm

    t0 = time.time()
    lms = MoranRes(ols, w, z=True)
    t1 = time.time()
    tf = t1 - t0
    runMoran = 'Moran test:\t\t\t%.5f seconds\n'%tf
    log.write(runMoran)
    print runMoran


    tff=time.time()
    tt = tff - ti
    total = 'Total final time:\t\t%.5f seconds\n'%tt
    log.write(total)
    print total
    log.close()
    return 'Done!'

#test_large_all(1100, 10, 'log.txt')

def test_large_sp(s, k, log):
    """
    Run and time OLS and spatial diagnostics
    ...

    Arguments
    ---------
    s       : int
              Side of the lattice from which to build the weights on
              n = s**2
    k       : int
              N of variables to include in regression
    log     : string
              Path of log file to append results to

    Returns
    -------
              Updates on-the-fly the log file with new timing results
    """
    log = open(log, 'a')
    n = 'n: %i\n'%s**2
    log.write(n)
    print n
    vars = 'k: %i\n'%k
    log.write(vars)
    print vars

    ti = time.time()
    t0 = time.time()
    data = Data(s**2, k, 'large', '../../../trunk/econometrics/', omit_w=True)
    t1 = time.time()
    tf = t1 - t0
    creDa = 'Create data:\t\t\t%.5f seconds\n'%tf
    log.write(creDa)
    print creDa

    t0 = time.time()
    w = ps.lat2W(s, s, rook=False)
    t1 = time.time()
    tf = t1 - t0
    creWe = 'Created Weights:\t\t%.5f seconds\n'%tf
    log.write(creWe)
    print creWe

    t0 = time.time()
    ols = BaseOLS(data.y, data.x)
    t1 = time.time()
    tf = t1 - t0
    runOls = 'Regression:\t\t\t%.5f seconds\n'%tf
    log.write(runOls)
    print runOls

    t0 = time.time()
    lms = LMtests(ols, w)
    t1 = time.time()
    tf = t1 - t0
    runLm = 'LM diagnostics:\t\t\t%.5f seconds\n'%tf
    log.write(runLm)
    print runLm

    t0 = time.time()
    lms = MoranRes(ols, w, z=True)
    t1 = time.time()
    tf = t1 - t0
    runMoran = 'Moran test:\t\t\t%.5f seconds\n'%tf
    log.write(runMoran)
    print runMoran


    tff=time.time()
    tt = tff - ti
    total = 'Total final time:\t\t%.5f seconds\n'%tt
    log.write(total)
    print total
    log.close()
    return 'Done!'

sizes = [150, 300, 450, 600, 750, 800, 900, 1000]
sizes = [1150, 1300, 1450, 1600, 1750, 1900, 2000]
for size in sizes:
    test_large_sp(1450, 10, 'log_sp.txt')


