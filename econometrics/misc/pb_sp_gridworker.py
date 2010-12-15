import sys
import numpy as np
import pickle
import struct
import os
import scipy.stats as stats
#Grid Worker:
"""
Evaluates the boundaries and gets the sum over R of the product of the cumulative distributions.
"""
args = sys.argv
R = int(args[1])
R_id = int(args[2])
infile = args[3]+'probit_sp.pkl'
#infile = 'Users/pedroamaral/Documents/Academico/GeodaCenter/python/SVN/spreg/trunk/econometrics/grid_in/probit_sp.pkl'
pkl_file = open(infile, 'rb')
data = pickle.load(pkl_file)
N = data[0]
V = data[1]
B = data[2]
pkl_file.close()
sumPhi = 0
for r in range(R):
    seed = abs(struct.unpack('i',os.urandom(4))[0])
    np.random.seed(seed)
    nn = np.zeros((N,1),float)
    vn = np.zeros((N,1),float)        
    sumbn = 0
    prodPhi = 1.0
    for i in range(N):
        n = -(i+1)
        vn[n] = 1.0*(V[n]-sumbn)/B[n,n]
        prodPhi = prodPhi * stats.norm.cdf(vn[n])
        if i<N-1:
            nn[n] = np.random.normal(0,1)
            while nn[n] >= vn[n]:
                nn[n] = np.random.normal(0,1)
            sumbn = np.dot(B[n-1:n,n:],nn[n:])
    if r == 5:
        test0 = prodPhi
    sumPhi += prodPhi
outfile = 'run_%s.pkl' %R_id
output = open(outfile, 'wb')
pickle.dump((sumPhi,test0), output, -1)
output.close()
