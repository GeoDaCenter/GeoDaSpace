"""
Script to test spreg functionality on Seattle dataset

NOTE: it tests functionality to be included in PySAL release 1.1 in Jan. 2011
"""

import pysal
import time
import numpy as np
from pysal.spreg.ols import BaseOLS as OLS
#from pysal.spreg.ols import OLS

def run_seattle(data_link, yvar, xvars, k=8, pts=True):
    print '\n\t\t\t### Seattle dataset benchmarking ###\n'

    ti = time.time()
    t0 = time.time()
    w_link = data_link
    if pts:
        w_link += '_pts'
    w_link += '.shp'
    w = pysal.knnW_from_shapefile(w_link, k=k)
    t1 = time.time()
    print 'Number of observations:\t\t%i\n'%w.n
    tf = t1 - t0
    print 'Shape reading and W creating:\t%.5f seconds'%tf

    t0 = time.time()
    nat = pysal.open(data_link + '.dbf')
    t1 = time.time()
    tf = t1 - t0
    print 'Loading data:\t\t\t%.5f seconds'%tf

    t0 = time.time()
    y = np.array([nat.by_col(yvar)]).T
    y = np.array(y, dtype=float)
    t1 = time.time()
    tf = t1 - t0
    print 'Creating dep var y:\t\t%.5f seconds'%tf

    t0 = time.time()
    x = map(nat.by_col, xvars)
    x = map(np.array, x)
    x = np.vstack(x)
    x = np.array(x.T, dtype=float)
    t1 = time.time()
    tf = t1 - t0
    print 'Creating indep vars x:\t\t%.5f seconds'%tf

    t0 = time.time()
    ols = OLS(y, x)
    #ols = OLS(y, x, name_y=yvar, name_x=xvars, name_ds=data_link.split('/')[-1], vm=True)
    t1 = time.time()
    tf = t1 - t0
    print 'Running OLS & diagnostics:\t%.5f seconds\n'%tf
    #return [ols]
    return [y, x]

'''
data_link = '/home/dani/AAA/LargeData/Seattle/parcel2000_city_resbldg99_larea'
xvars = ['GRADE', 'BEDROOM', 'BASEAREA', 'FULLBATH', 'YEARBLT', 'CONDTION']
yvar = 'LIVEAREA'
model = run_seattle(data_link, yvar, xvars)

'''
data_link = '/home/dani/AAA/LargeData/Seattle/all97_Aug08_bis_resarea_tract'
xvars = ['firplace', 'bricksto', 'bathfull', 'lake1k', 'park250', 'stories2', \
    'viewall', 'nowhite', 'pcty65', 'pctcol', 'age', 'higrade', 'mindist_li', \
    'sqfttotl_r', 'sqftlive_r', 'apex', 'wx_agep', 'wx_conpr', 'wx_sfhom', \
    'wx_duplx']
yvar = 'lnprice'
model = run_seattle(data_link, yvar, xvars, pts=False)

