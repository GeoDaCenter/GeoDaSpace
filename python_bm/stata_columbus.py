'''
Script to replicate steps in ../stata_bm/columbus.do
'''

import pysal as ps
import numpy as np
import econometrics as spreg

w = ps.open('../../../trunk/econometrics/examples/columbus.gal').read()
w.transform = 'r'

db = ps.open('../../../trunk/econometrics/examples/columbus.dbf')

crime = np.array(db.by_col('CRIME'))
w_crime = ps.lag_spatial(w, crime)

