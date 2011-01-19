"""
Script to create large datasets and benchmark times
"""


import pysal as ps
import time
import numpy as np
from econometrics.ols import OLS
from econometrics.testing_utils import Test_Data as Data

print '\n\t\t\t### Large simulated dataset benchmarking ###\n'

s = 1000 # Side of the lattice
k = 10
print 'n: %i'%s**2
print 'k: %i'%k

ti = time.time()
t0 = time.time()
data = Data(s**2, k, 'large', '../../../trunk/')
t1 = time.time()
tf = t1 - t0
print 'Simulated data set created:\t%.5f seconds'%tf

t0 = time.time()
w = ps.lat2W(s, s, rook=False)
t1 = time.time()
tf = t1 - t0
print 'Weights object created:\t\t%.5f seconds'%tf

ols = 

tff=time.time()
tt = tff - ti
print 'Total final time:\t%.5f seconds'%tt
