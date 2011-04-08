"""
Script to test the Spatial Diagnostics module on spreg
"""

import numpy as np
import pysal, time

trunk = '../../../trunk/econometrics/'

# Data Loading
from econometrics.testing_utils import Test_Data 
from pysal.spreg.diagnostics_sp import  spDcache
from econometrics.ak import akTest, akTest_legacy, AKtest
from econometrics.twosls_sp import STSLS as STSLS

## 100 obs
data = Test_Data(100, 4, 'medium', trunk)

# Moran's I

# LM tests

# AK tests

############ Large sim ############
s = 1000
print '### AK benchmarking ###'
print 'n: ', s*s
print 'k: 5'
data = np.random.random((s*s, 6))
y, X = data[:, 0:1], data[:, 1:]
w = pysal.lat2W(s, s)
print 'W: rook lattice'
###################################

iv = STSLS(y, X, w, w_lags=1)

print '\n\t### SPATIAL 2SLS results from PySAL ###'
#print '###Betas:\n', iv.betas

t0 = time.time()
ak = AKtest(iv, w, case='gen')
t1 = time.time()
print 'Time: ', t1-t0
'''
cache = spDcache(iv, w)
ak = akTest(iv, w, cache)
'''
print '###AK test:\t', ak.ak, ak.p
