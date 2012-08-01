"""
Example using OLS with regimes in PySAL.
Obs: added to this folder just to ensure there will be no
path-related issues at the workshop.
"""

import numpy as np
import pysal
from ols_regimes import OLS_Regimes

db = pysal.open('examples/columbus.dbf','r')
y_var = 'CRIME'
y = np.array([db.by_col(y_var)]).T
x_var = ['INC','HOVAL']
x = np.array([db.by_col(name) for name in x_var]).T
regimes = db.by_col('NSA')
w = pysal.open('examples/columbus.gal','r').read()
w.transform='r'

olsr = OLS_Regimes(y, x, regimes, w=w, constant_regi='many', spat_diag=True,
                   nonspat_diag=False, moran=True, name_y=y_var,
                   name_x=x_var, name_ds='columbus',
                   name_w='columbus.gal')

print olsr.summary

"""  
Parameters
----------
constant_regi: [False, 'one', 'many']
               Ignored if regimes=False. Constant option for regimes.
               Switcher controlling the constant term setup. It may take
               the following values:
                
                 *  False: no constant term is appended in any way
                 *  'one': a vector of ones is appended to x and held
                           constant across regimes
                 * 'many': a vector of ones is appended to x and considered
                           different per regime

robust       : string
               If 'white', then a White consistent estimator of the
               variance-covariance matrix is given.  If 'hac', then a
               HAC consistent estimator of the variance-covariance
               matrix is given. Default set to None. 

gwk          : pysal W object
               Kernel spatial weights needed for HAC estimation. Note:
               matrix must have ones along the main diagonal.
"""
