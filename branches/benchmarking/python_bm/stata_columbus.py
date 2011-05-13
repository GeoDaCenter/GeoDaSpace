'''
Script to replicate steps in ../stata_bm/columbus.do
Script to replicate steps in ../r_bm/stata_columbus.r
'''

import pysal as ps
import numpy as np
from econometrics.spError import GM_Error, BaseGM_Endog_Error_Hom
from econometrics.twosls_sp import BaseGM_Lag as GM_Lag
from econometrics.spError import BaseGM_Combo as GM_Combo
from econometrics.spHetErr import BaseGM_Error_Het as BaseGM_Error_Het

w = ps.open('../../../trunk/econometrics/examples/columbus.gal').read()
w.transform = 'r'
db = ps.open('../../../trunk/econometrics/examples/columbus.dbf')

crime = np.array([db.by_col('CRIME')]).T
inc = np.array([db.by_col('INC')]).T
hoval = np.array([db.by_col('HOVAL')]).T
discbd = np.array([db.by_col('DISCBD')]).T
#w_crime = ps.lag_spatial(w, crime)

x = np.hstack((inc, crime))

# OLS (matches both)
#model = ps.spreg.ols.OLS(hoval, x)

# GM Error (matches R, not STATA)
#model = GM_Error(hoval, x, w) # This model is not implemented in STATA's 'spivreg'

# GM_hom
#model = BaseGM_Endog_Error_Hom(hoval, inc, w, crime, discbd)

# 2SLS Lag 
#model = GM_Lag(hoval, x, w, w_lags=2)
#model = GM_Lag(hoval, x, w, w_lags=2, robust="White")
#model = GM_Lag(hoval, inc, yend=crime, q=discbd, w=w, w_lags=2)
#model = GM_Lag(hoval, inc, yend=crime, q=discbd, w=w, w_lags=2, robust="White")

# GM_Combo
#model = GM_Combo(hoval, x, w, w_lags=2)

# GM Error Het
model = BaseGM_Error_Het(hoval, x, w)


print '##### Betas #####'
print model.betas
#   print '### Std Devs ###'
print map(np.sqrt, model.vm.diagonal())
#print map(np.sqrt, model.vm.diagonal() * model.n/(model.n-model.k))
#   print '### VC Matrix Omega ###'
#   print model.vm
#print '### Initial estimate of lambda ###'
#print model.lambda1
#   print '### Initial estimate of betas ###'
#   print model.tsls.betas

