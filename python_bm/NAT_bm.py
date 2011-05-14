"""
Script to test spreg functionality on NAT dataset

NOTE: it tests functionality to be included in PySAL release 1.1 in Jan. 2011
"""

import pysal
import time
import numpy as np
from econometrics.ols import OLS, BaseOLS
from econometrics.twosls_sp import BaseGM_Lag as GM_Lag
from econometrics.spHetErr import BaseGM_Error_Het as BaseGM_Error_Het

data_link = '../../../trunk/econometrics/examples/NAT.'

print '\n\t\t\t### NAT dataset benchmarking ###\n'

ti = time.time()
t0 = time.time()
#w = pysal.queen_from_shapefile(data_link + 'shp')
w = pysal.open('../../../trunk/econometrics/examples/NAT_queen.gal').read()
w.transform='r'
t1 = time.time()
#print 'Number of observations:\t\t%i\n'%w.n
tf = t1 - t0
#print 'Shape reading and W creating:\t%.5f seconds'%tf

t0 = time.time()
nat = pysal.open(data_link + 'dbf')
t1 = time.time()
tf = t1 - t0
#print 'Loading data:\t\t\t%.5f seconds'%tf

t0 = time.time()
y = np.array([nat.by_col('HR90')]).T
t1 = time.time()
tf = t1 - t0
#print 'Creating dep var y:\t\t%.5f seconds'%tf

t0 = time.time()
xvars = ['RD90', 'MA90', 'DV90', 'BLK90']
xvars = ['MA90', 'DV90']
x = map(nat.by_col, xvars)
x = map(np.array, x)
x = np.vstack(x)
x = x.T
t1 = time.time()
tf = t1 - t0
#print 'Creating indep vars x:\t\t%.5f seconds'%tf

t0 = time.time()
model = BaseGM_Error_Het(y, x, w)
#model = GM_Lag(y, x, w, w_lags=2)
#model = BaseOLS(y, x)
t1 = time.time()
tf = t1 - t0
#print 'Running OLS & diagnostics:\t%.5f seconds\n'%tf

print '# Betas #'
print model.betas

