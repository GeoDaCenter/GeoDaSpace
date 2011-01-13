"""
Script to test spreg functionality on NAT dataset

NOTE: it tests functionality to be included in PySAL release 1.1 in Jan. 2011
"""

import pysal
import time
import numpy as np

data_link = '../../../trunk/econometrics/examples/NAT.'

print '\n\t\t\t### NAT dataset benchmarking ###\n'

ti = time.time()
t0 = time.time()
w = pysal.queen_from_shapefile(data_link + 'shp')
t1 = time.time()
print 'Number of observations:\t\t%i\n'%w.n
tf = t1 - t0
print 'Shape reading and W creating:\t%.5f seconds'%tf

t0 = time.time()
nat = pysal.open(data_link + 'dbf')
t1 = time.time()
tf = t1 - t0
print 'Loading data:\t\t\t%.5f seconds'%tf

t0 = time.time()
y = np.array(nat.by_col('HR60'))
t1 = time.time()
tf = t1 - t0
print 'Creating dep var y:\t\t%.5f seconds'%tf

t0 = time.time()

t1 = time.time()
tf = t1 - t0
print 'Creating indep vars x:\t\t%.5f seconds'%tf

