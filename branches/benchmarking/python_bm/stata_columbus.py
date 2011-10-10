'''
Script to replicate steps in ../stata_bm/columbus.do
Script to replicate steps in ../r_bm/stata_columbus.r
'''

import pysal as ps
import numpy as np
#from econometrics.spError import GM_Error, BaseGM_Endog_Error_Hom
from econometrics.twosls_sp import BaseGM_Lag as GM_Lag
#from econometrics.spError import BaseGM_Combo as GM_Combo
from econometrics.error_sp_het import BaseGM_Error_Het as BaseGM_Error_Het
from econometrics.error_sp_het import BaseGM_Endog_Error_Het as BaseGM_Endog_Error_Het
from econometrics.error_sp_het import BaseGM_Combo_Het as BaseGM_Combo_Het

#w = ps.open('../../../trunk/econometrics/examples/columbus.gal').read()
w = ps.open('C:/Users/Pedro/Documents/Academico/GeodaCenter/python/SVN/spreg/trunk/econometrics/examples/columbus.gal').read()
w.transform = 'r'
#db = ps.open('../../../trunk/econometrics/examples/columbus.dbf')
db = ps.open('C:/Users/Pedro/Documents/Academico/GeodaCenter/python/SVN/spreg/trunk/econometrics/examples/columbus.dbf')

crime = np.array([db.by_col('CRIME')]).T
inc = np.array([db.by_col('INC')]).T
hoval = np.array([db.by_col('HOVAL')]).T
discbd = np.array([db.by_col('DISCBD')]).T
#w_crime = ps.lag_spatial(w, crime)

x = np.hstack((hoval, inc))

# OLS (matches both)
#model = ps.spreg.ols.OLS(crime, x)


# 2SLS Lag 
#model = GM_Lag(crime, x, w, w_lags=2)
#model = GM_Lag(crime, x, w, w_lags=2, robust="White")
#model = GM_Lag(crime, inc, yend=hoval, q=discbd, w=w, w_lags=2)
#model = GM_Lag(crime, inc, yend=hoval, q=discbd, w=w, w_lags=2, robust="White")



# GM Error (matches R, not STATA)
#model = GM_Error(crime, x, w) # This model is not implemented in STATA's 'spivreg'
# GM Combo
#model = GM_Combo(crime, x, w, w_lags=2)



# GM Error Hom
#model = BaseGM_Endog_Error_Hom(crime, inc, w, hoval, discbd)
# GM Combo Hom
#model = BaseGM_Endog_Combo_Hom(crime, inc, w, hoval, discbd)



# GM Error Het
#model = BaseGM_Error_Het(crime, x, w) #to match Stata the following code is required:
ones = np.ones(crime.shape)
model = BaseGM_Endog_Error_Het(crime, ones, x, x, w, step1c=False, constant=False)
'''
self.x = reg.z
self.y = reg.y
self.n, self.k = reg.n, reg.k
self.betas = reg.betas
self.vm = reg.vm
self.u = reg.u
self.predy = reg.predy
self._cache = {}
'''
# GM Error Het with user-defined endog
#model = BaseGM_Endog_Error_Het(crime, inc, hoval, discbd, w, step1c=False)
# GM Combo Het
#model = BaseGM_Combo_Het(crime, x, w=w, w_lags=2, step1c=False)
# GM Combo Het with user-defined endog
#model = BaseGM_Combo_Het(crime, inc, w=w, yend=hoval, q=discbd, w_lags=2, step1c=False)


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

