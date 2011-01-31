"""
Script to test spreg functionality on Seattle dataset

NOTE: it tests functionality to be included in PySAL release 1.1 in Jan. 2011
"""

import pysal
import time
import numpy as np
from pysal.spreg.ols import BaseOLS as OLS
from pysal.spreg.diagnostics_sp import LMtests, MoranRes
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
    tp0 = time.time()
    x = map(nat.by_col, xvars)
    tp1 = time.time()
    tp = tp1 - tp0
    print 'by_col: %f'%tp
    x = np.array(x).T
    #x = map(np.array, x)
    #x = np.vstack(x)
    #x = np.array(x.T, dtype=float)
    t1 = time.time()
    arr = t1 - tp1
    print 'convert to array: %f'%arr
    tf = t1 - t0
    print 'Creating indep vars x:\t\t%.5f seconds'%tf
   #t0 = time.time()
   #x = map(nat.by_col, xvars)
   #x = map(np.array, x)
   #x = np.vstack(x)
   #x = np.array(x.T, dtype=float)
   #t1 = time.time()
   #tf = t1 - t0
   #print 'Creating indep vars x:\t\t%.5f seconds'%tf

    t0 = time.time()
    ols = OLS(y, x)
    t1 = time.time()
    tf = t1 - t0
    runOls = 'Regression:\t\t\t%.5f seconds\n'%tf
    print runOls

    t0 = time.time()
    lms = LMtests(ols, w)
    t1 = time.time()
    tf = t1 - t0
    runLm = 'LM diagnostics:\t\t\t%.5f seconds\n'%tf
    print runLm

    t0 = time.time()
    moran = MoranRes(ols, w, z=True)
    t1 = time.time()
    tf = t1 - t0
    runMoran = 'Moran test:\t\t\t%.5f seconds\n'%tf
    print runMoran

   #t0 = time.time()
   #ols = OLS(y, x)
   #ols = OLS(y, x, name_y=yvar, name_x=xvars, name_ds=data_link.split('/')[-1], vm=True)
   #t1 = time.time()
   #tf = t1 - t0
   #print 'Running OLS & diagnostics:\t%.5f seconds\n'%tf

    tff=time.time()
    tt = tff - ti
    total = 'Total final time:\t\t%.5f seconds\n'%tt
    print total

    return [ols]

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
'''

