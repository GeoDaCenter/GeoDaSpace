"""
Script to test spreg functionality on Seattle dataset

NOTE: it tests functionality to be included in PySAL release 1.1 in Jan. 2011
"""

import pysal
import time
import numpy as np
from econometrics.ols import OLS

data_link = '/home/dani/AAA/LargeData/Seattle/parcel2000_city_resbldg99clean'

print '\n\t\t\t### Seattle dataset benchmarking ###\n'

ti = time.time()
t0 = time.time()
w = pysal.open(data_link + '_pts.gwt').read()
t1 = time.time()
print 'Number of observations:\t\t%i\n'%w.n
tf = t1 - t0
print 'Shape reading and W creating:\t%.5f seconds'%tf

t0 = time.time()
nat = pysal.open(data_link + '.dbf')
t1 = time.time()
tf = t1 - t0
print 'Loading data:\t\t\t%.5f seconds'%tf

"""
t0 = time.time()
y = np.array([nat.by_col('HR90')]).T
t1 = time.time()
tf = t1 - t0
print 'Creating dep var y:\t\t%.5f seconds'%tf

t0 = time.time()
xvars = ['RD90', 'MA90', 'DV90', 'BLK90']
xvars = ['MA90', 'DV90']
x = map(nat.by_col, xvars)
x = map(np.array, x)
x = np.vstack(x)
x = x.T
t1 = time.time()
tf = t1 - t0
print 'Creating indep vars x:\t\t%.5f seconds'%tf

t0 = time.time()
ols = OLS(y, x, w, name_y='HR60', name_x=xvars, name_ds='NAT', vm=True)
t1 = time.time()
tf = t1 - t0
print 'Running OLS & diagnostics:\t%.5f seconds\n'%tf

"""
