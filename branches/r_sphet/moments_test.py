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

fo = open('moments_runPy.txt', 'w')
fo.write('Time elapsed to run gmmS.Moments in different datasets\n')
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

        print 'Running Moments...'
        t0 = time()
        m = Moments(w, u)
        t1 = time()

        line = '%s,%i,%f\n'%(k, n, t1-t0)
        fo.write(line)
fo.close()

