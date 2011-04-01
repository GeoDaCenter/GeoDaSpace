"""
Script to create large datasets and benchmark times
"""

import pysal as ps
import time
import numpy as np
from pysal.spreg.ols import OLS
from pysal.spreg.ols import BaseOLS
from opt_diagnostics_sp import LMtests, MoranRes
#from pysal.spreg.diagnostics_sp import LMtests, MoranRes
from econometrics.spError import BaseGMSWLS, GMSWLS
from econometrics.twosls_sp import BaseSTSLS, STSLS
from econometrics.testing_utils import Test_Data as Data
from workbench import Model_Inputs

print '\n\t\t\t### Large simulated dataset benchmarking ###\n'

def test_large_olsSPd(s, k, log=None, base=False, sw=False, a=True):
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
    a       : boolean
              Switcher to append log to existing file or just print to
              terminal

    Returns
    -------
              Updates on-the-fly the log file with new timing results
    """
    if a:
        log = open(log, 'a')
    n = 'n: %i\n'%s**2
    if a:
        log.write(n)
    print n
    vars = 'k: %i\n'%k
    if a:
        log.write(vars)
    print vars

    ti = time.time()
    t0 = time.time()
    data = Data(s**2, k, 'large', '../../../trunk/econometrics/', omit_w=True)
    t1 = time.time()
    tf = t1 - t0
    creDa = 'Create data:\t\t\t%.5f seconds\n'%tf
    if a:
        log.write(creDa)
    print creDa

    t0 = time.time()
    if sw:
        w = ps.weights.lat2SW(s, s, criterion='queen')
        w.n = s**2
        w.sparse = w
        w.s0 = np.sum(w.data)

        t = w.T + w
        t2 = t.multiply(t) # element-wise square
        w.s1 = t2.sum()/2.

    else:
        w = ps.lat2W(s, s, rook=False)
    t1 = time.time()
    tf = t1 - t0
    creWe = 'Created Weights:\t\t%.5f seconds\n'%tf
    if a:
        log.write(creWe)
    print creWe

    t0 = time.time()
    if base:
        print 'Running barebone version of OLS'
        ols = BaseOLS(data.y, data.x)
    else:
        print 'Running full end-user version of OLS'
        ols = OLS(data.y, data.x)
    t1 = time.time()
    tf = t1 - t0
    runOls = 'Regression:\t\t\t%.5f seconds\n'%tf
    if a:
        log.write(runOls)
    print runOls

    '''
    t0 = time.time()
    lms = LMtests(ols, w)
    t1 = time.time()
    tf = t1 - t0
    runLm = 'LM diagnostics:\t\t\t%.5f seconds\n'%tf
    if a:
        log.write(runLm)
    print runLm
    '''

    t0 = time.time()
    moran = MoranRes(ols, w, z=True)
    t1 = time.time()
    tf = t1 - t0
    runMoran = 'Moran test:\t\t\t%.5f seconds\n'%tf
    if a:
        log.write(runMoran)
    print runMoran

    tff=time.time()
    tt = tff - ti
    total = 'Total final time:\t\t%.5f seconds\n'%tt
    if a:
        log.write(total)
    print total
    if a:
        log.close()
    return [ols, lms, moran]

def test_large_GMSWLS(s, k, log=None, base=False, sw=False, a=True):
    """
    Run and time GMSWLS 
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
    a       : boolean
              Switcher to append log to existing file or just print to
              terminal

    Returns
    -------
              Updates on-the-fly the log file with new timing results
    """
    if not log:
        a = False
        print 'Not appending to any file'
    if a:
        log = open(log, 'a')
    n = 'n: %i\n'%s**2
    if a:
        log.write(n)
    model = 'Model: GMSWLS'
    if a:
        log.write(model)
    print n
    vars = 'k: %i\n'%k
    if a:
        log.write(vars)
    print vars

    ti = time.time()
    t0 = time.time()
    data = Model_Inputs(s**2, intercept=False, x_means=[0.]*(k+1),
            x_std_devs=[1.]*(k+1), xs=(k+1)) 
    y, x = data.x[:, 0:1], data.x[:, 1:]
    t1 = time.time()
    tf = t1 - t0
    creDa = 'Create data:\t\t\t%.5f seconds\n'%tf
    if a:
        log.write(creDa)
    print creDa

    t0 = time.time()
    if sw:
        w = ps.weights.lat2SW(s, s, criterion='queen')
        w.n = s**2
        w.sparse = w
        w.s0 = np.sum(w.data)

        t = w.T + w
        t2 = t.multiply(t) # element-wise square
        w.s1 = t2.sum()/2.

    else:
        w = ps.lat2W(s, s, rook=False)
    t1 = time.time()
    tf = t1 - t0
    creWe = 'Created Weights:\t\t%.5f seconds\n'%tf
    if a:
        log.write(creWe)
    print creWe

    t0 = time.time()
    if base:
        print 'Running barebone version of GMSWLS'
        gmswls = BaseGMSWLS(y, x, w)
    else:
        print 'Running full end-user version of GMSWLS'
        gmswls = GMSWLS(y, x, w)
    t1 = time.time()
    tf = t1 - t0
    runGmswls = 'Regression:\t\t\t%.5f seconds\n'%tf
    if a:
        log.write(runGmswls)
    print runGmswls

    tff=time.time()
    tt = tff - ti
    total = 'Total final time:\t\t%.5f seconds\n'%tt
    if a:
        log.write(total)
    print total
    if a:
        log.close()
    return gmswls

def test_large_STSLS(s, k, log=None, base=False, sw=False, a=True):
    """
    Run and time STSLS 
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
    a       : boolean
              Switcher to append log to existing file or just print to
              terminal

    Returns
    -------
              Updates on-the-fly the log file with new timing results
    """
    if not log:
        a = False
        print 'Not appending to any file'
    if a:
        log = open(log, 'a')
    n = 'n: %i\n'%s**2
    if a:
        log.write(n)
    model = 'Model: STSLS'
    if a:
        log.write(model)
    print n
    vars = 'k: %i\n'%k
    if a:
        log.write(vars)
    print vars

    ti = time.time()
    t0 = time.time()
    data = Model_Inputs(s**2, intercept=False, x_means=[0.]*(k+1),
            x_std_devs=[1.]*(k+1), xs=(k+1)) 
    y, x = data.x[:, 0:1], data.x[:, 1:]
    t1 = time.time()
    tf = t1 - t0
    creDa = 'Create data:\t\t\t%.5f seconds\n'%tf
    if a:
        log.write(creDa)
    print creDa

    t0 = time.time()
    if sw:
        w = ps.weights.lat2SW(s, s, criterion='queen')
        w.n = s**2
        w.sparse = w
        w.s0 = np.sum(w.data)

        t = w.T + w
        t2 = t.multiply(t) # element-wise square
        w.s1 = t2.sum()/2.

    else:
        w = ps.lat2W(s, s, rook=False)
    t1 = time.time()
    tf = t1 - t0
    creWe = 'Created Weights:\t\t%.5f seconds\n'%tf
    if a:
        log.write(creWe)
    print creWe

    t0 = time.time()
    if base:
        print 'Running barebone version of STSLS'
        stsls = BaseSTSLS(y, x, w)
    else:
        print 'Running full end-user version of STSLS'
        stsls = STSLS(y, x, w)
    t1 = time.time()
    tf = t1 - t0
    runStsls = 'Regression:\t\t\t%.5f seconds\n'%tf
    if a:
        log.write(runStsls)
    print runStsls

    tff=time.time()
    tt = tff - ti
    total = 'Total final time:\t\t%.5f seconds\n'%tt
    if a:
        log.write(total)
    print total
    if a:
        log.close()
    return stsls

def test_large_sp_models(s, k, log=None, base=False, sw=False, a=True):
    """
    Run and time STSLS 
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
    a       : boolean
              Switcher to append log to existing file or just print to
              terminal

    Returns
    -------
              Updates on-the-fly the log file with new timing results
    """
    if not log:
        a = False
        print 'Not appending to any file'
    if a:
        log = open(log, 'a')
    n = 'n: %i\n'%s**2
    if a:
        log.write(n)
    model = 'Model: STSLS'
    if a:
        log.write(model)
    print n
    vars = 'k: %i\n'%k
    if a:
        log.write(vars)
    print vars

    ti = time.time()
    t0 = time.time()
    data = Model_Inputs(s**2, intercept=False, x_means=[0.]*(k+1),
            x_std_devs=[1.]*(k+1), xs=(k+1)) 
    y, x = data.x[:, 0:1], data.x[:, 1:]
    t1 = time.time()
    tf = t1 - t0
    creDa = 'Create data:\t\t\t%.5f seconds\n'%tf
    if a:
        log.write(creDa)
    print creDa

    t0 = time.time()
    if sw:
        w = ps.weights.lat2SW(s, s, criterion='queen')
        w.n = s**2
        w.sparse = w
        w.s0 = np.sum(w.data)

        t = w.T + w
        t2 = t.multiply(t) # element-wise square
        w.s1 = t2.sum()/2.

    else:
        w = ps.lat2W(s, s, rook=False)
    t1 = time.time()
    tf = t1 - t0
    creWe = 'Created Weights:\t\t%.5f seconds\n'%tf
    if a:
        log.write(creWe)
    print creWe

    t0 = time.time()
    if base:
        print 'Running barebone version of GMSWLS'
        gmswls = BaseGMSWLS(y, x, w)
    else:
        print 'Running full end-user version of GMSWLS'
        gmswls = GMSWLS(y, x, w)
    t1 = time.time()
    tf = t1 - t0
    runGmswls = 'GMSWLS:\t\t\t%.5f seconds\n'%tf
    if a:
        log.write(runGmswls)
    print runGmswls

    t0 = time.time()
    if base:
        print 'Running barebone version of STSLS'
        stsls = BaseSTSLS(y, x, w)
    else:
        print 'Running full end-user version of STSLS'
        stsls = STSLS(y, x, w)
    t1 = time.time()
    tf = t1 - t0
    runStsls = 'STSLS:\t\t\t%.5f seconds\n'%tf
    if a:
        log.write(runStsls)
    print runStsls

    tff=time.time()
    tt = tff - ti
    total = 'Total final time:\t\t%.5f seconds\n'%tt
    if a:
        log.write(total)
    print total
    if a:
        log.close()
    return stsls

sizes = [150, 300, 450]
#sizes = [500, 750, 1000, 1150, 1300, 1450, 1600]
#sizes = [1750, 1900, 2000, 2050, 2100]
#sizes = [1750]

for side in sizes:
    #gmswls = test_large_GMSWLS(side, 10, log='logs/gmswls_py.log', a=True, base=True, sw=False)
    #stsls = test_large_STSLS(side, 10, log='logs/stsls_py.log', a=True, base=True, sw=False)
    sp_models = test_large_sp_models(side, 10, log='logs/sp_models.log', a=True, base=True, sw=False)


