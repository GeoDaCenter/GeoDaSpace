'''
Debug hom with Lee's data
'''

import pysal as ps
import numpy as np
#from econometrics.spError import GM_Error, BaseGM_Endog_Error_Hom
from econometrics.twosls_sp import BaseGM_Lag as GM_Lag
#from econometrics.spError import BaseGM_Combo as GM_Combo
from econometrics.error_sp_het import BaseGM_Error_Het as BaseGM_Error_Het
from econometrics.error_sp_hom import BaseGM_Endog_Error_Hom as BaseGM_Endog_Error_Hom
from econometrics.error_sp_hom import GM_Error_Hom, GM_Endog_Error_Hom
from econometrics.error_sp_het import BaseGM_Endog_Error_Het as BaseGM_Endog_Error_Het
from econometrics.error_sp_het import BaseGM_Combo_Het as BaseGM_Combo_Het
from econometrics.error_sp_hom import BaseGM_Combo_Hom as BaseGM_Combo_Hom
from econometrics.twosls import TSLS

w = ps.open('lee_data/crc.gal').read()
w.transform = 'r'
db = ps.open('lee_data/crc.dbf')
y = np.array(db.by_col('crc_scrn01'.upper())).reshape((w.n, 1))
var_names = ['age7584', 'age85plus', 'dual_esrd', 'race_black',
        'race_hispa', 'raceother', 'mover01_05', 'sig_col_di', 'cont_care',
        'st_perpov', 
        #'c_all_hmo', 
        #'age65_exer'
        ]
yend = np.array(db.by_col['C_ALL_HMO']).reshape((w.n, 1))
q = np.array(db.by_col['AGE65_EXER']).reshape((w.n, 1))
vars = [db.by_col[i.upper()] for i in var_names]
#x = np.array(zip(vars))
x = np.array(vars).T

# OLS (matches both)
#model = ps.spreg.ols.OLS(y, x)


# 2SLS Lag 
#model = TSLS(hoval, inc, yend=crime, q=discbd, sig2n_k=True)
#model = GM_Lag(hoval, x, w=w, w_lags=2)
#model = GM_Lag(crime, x, w, w_lags=2, robust="White")
#model = GM_Lag(crime, inc, yend=hoval, q=discbd, w=w, w_lags=2)
#model = GM_Lag(crime, inc, yend=hoval, q=discbd, w=w, w_lags=2, robust="White")



# GM Error (matches R, not STATA)
#model = GM_Error(crime, x, w) # This model is not implemented in STATA's 'spivreg'
# GM Combo
#model = GM_Combo(crime, x, w, w_lags=2)



# GM Error Hom
ones = np.ones(y.shape)
#model = GM_Endog_Error_Hom(y, ones, x, x, w)
model = GM_Endog_Error_Hom(y, x, w=w, yend=yend, q=q, name_x=var_names, A1='hom_sc')
#model = GM_Error_Hom(y, x, w, name_x=var_names, A1='hom_sc')
print model.summary
# GM Combo Hom
#model = BaseGM_Combo_Hom(hoval, x, w=w, w_lags=2)
#model = BaseGM_Combo_Hom(hoval, x, w=w, w_lags=2)
#model = BaseGM_Endog_Error_Hom(hoval, x, yend=crime, q=discbd, w=w,\
#        A1='hom_sc')


# GM Error Het
#model = BaseGM_Error_Het(crime, x, w) #to match Stata the following code is required:
#ones = np.ones(crime.shape)
#model = BaseGM_Endog_Error_Het(crime, ones, x, x, w, step1c=False, constant=False)
# GM Error Het with user-defined endog
#model = BaseGM_Endog_Error_Het(crime, inc, hoval, discbd, w, step1c=False)
# GM Combo Het
#model = BaseGM_Combo_Het(hoval, x, w=w, w_lags=2, step1c=False)
# GM Combo Het with user-defined endog
#model = BaseGM_Combo_Het(crime, x, w=w, yend=hoval, q=discbd, w_lags=2, step1c=False)

# GM Endog Error Het
#model = BaseGM_Endog_Error_Het(hoval, inc, w, crime, discbd)


#print '##### Betas #####'
#print model.betas
#   print '### Std Devs ###'
#print map(np.sqrt, model.vm.diagonal())

#for i,j in zip(model.betas, map(np.sqrt, model.vm.diagonal())):
#    print i, '\t', j
#print map(np.sqrt, model.vm.diagonal() * model.n/(model.n-model.k))
#print '### VC Matrix Omega ###'
#print model.vm
#print '### Initial estimate of lambda ###'
#print model.lambda1
#   print '### Initial estimate of betas ###'
#   print model.tsls.betas

