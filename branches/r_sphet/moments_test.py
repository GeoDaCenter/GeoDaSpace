"""Script to test gmmS in python Vs. sphet in R"""

from gmmS import Moments
from spHetErr import get_S, get_A1
import numpy as np
import pysal

u = pysal.open('/Users/dreamessence/repos/spreg/trunk/examples/n100_stdnorm_vars6.csv','r')
u = np.array([u.by_col('varA')]).T
w = pysal.open('/Users/dreamessence/repos/spreg/trunk/examples/w_rook_n100_order1_k4.gal').read()
w.transform='r'
w.S = get_S(w)
w.A1 = get_A1(w.S) 
m = Moments(w, u)


