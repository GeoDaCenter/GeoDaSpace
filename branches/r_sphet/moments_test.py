"""Script to test gmmS in python Vs. sphet in R"""

from gmmS import Moments
from testing_utils import Test_Data
from spHetErr import get_S, get_A1
from time import time
import numpy as np
import pysal

dataF = '/Users/dani/repos/spreg/trunk/'

ns = [100, 10000, 1000000]
ns = [100, 10000]
vars = {100:'n100_stdnorm_vars6.csv', 10000:'n10000_stdnorm_vars6.csv', 1000000:'n1000000_stdnorm_vars6.csv'}
ks = ['small', 'medium', 'large']
#ks = ['medium']

"""
fo = open('moments_runPy.txt', 'w')
fo.write('Time elapsed (secs) to run gmmS.Moments in different datasets\n')
for n in ns:
    print 'N = %i'%n
    for k in ks:
        print '  k = %s'%k
        data = Test_Data(n=n, k=k, folder=dataF)
        u = pysal.open(dataF+'examples/'+vars[n])
        u = np.array([u.by_col('varA')]).T
        w = data.w
        w.transform = 'r'
        w.S = get_S(w)
        w.A1 = get_A1(w.S) 

        print '   Running Moments...'
        t0 = time()
        m = Moments(w, u)
        t1 = time()

        line = '%s,%i,%f\n'%(k, n, t1-t0)
        fo.write(line)

# 'small', n = 1000000
"""
n, k = 1000000, 'small'
print 'N = %i'%n
print '  k = %s'%k
ti0 = time()
data = Test_Data(n=n, k=k, folder=dataF)
ti1 = time()
print '   w loaded: %f seconds'%(ti1-ti0)
u = pysal.open(dataF+'examples/'+vars[n])
u = np.array([u.by_col('varA')]).T
ti2 = time()
print '   u created: %f'%(ti2-ti1)
w = data.w
w.transform = 'r'
ti3 = time()
print '   w transformed: %f seconds'%(ti3-ti2)
w.S = get_S(w)
ti4 = time()
print '   w to sparse: %f seconds'%(ti4-ti3)
w.A1 = get_A1(w.S) 
ti5 = time()
print '   w has A1: %f seconds'%(ti5-ti4)

print '   Running Moments...'
t0 = time()
m = Moments(w, u)
t1 = time()
print 'Moments in %f seconds'%(t1-t0)

"""
line = '%s,%i,%f\n'%(k, n, t1-t0)
fo.append(line)


fo.close()
"""

