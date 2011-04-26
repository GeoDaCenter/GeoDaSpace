'''
Script to replicate steps in ../stata_bm/columbus.do
'''

import pysal as ps
import numpy as np
from econometrics.spError import GM_Error
from econometrics.twosls_sp import BaseGM_Lag as GM_Lag

w = ps.open('../../../trunk/econometrics/examples/columbus.gal').read()
w.transform = 'r'
db = ps.open('../../../trunk/econometrics/examples/columbus.dbf')

crime = np.array([db.by_col('CRIME')]).T
inc = np.array([db.by_col('INC')]).T
hoval = np.array([db.by_col('HOVAL')]).T
#w_crime = ps.lag_spatial(w, crime)

x = np.hstack((inc, crime))

# OLS (matches both)
#model = ps.spreg.ols.OLS(hoval, x)

# GM Error (matches R, not STATA)
#model = GM_Error(hoval, x, w)

# 2SLS Lag (does not match R or STATA, which both match)
model = GM_Lag(hoval, x, w, w_lags=2)

print model.betas
