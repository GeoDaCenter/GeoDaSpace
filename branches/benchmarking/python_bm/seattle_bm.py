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
    if k=='min':
        print 'Calculating Min. Threshold'
        thres = pysal.min_threshold_dist_from_shapefile(w_link)
        print 'Computing distance weights'
        w = pysal.threshold_binaryW_from_shapefile(w_link, thres)
    else:
        w = pysal.knnW_from_shapefile(w_link, k=k)
    t1 = time.time()
    print 'Number of observations:\t\t%i'%w.n
    print 'Number of variables:\t\t%i\n'%k
    tf = t1 - t0
    print 'Shape reading and W creating:\t%.5f seconds'%tf
    t0 = time.time()
    nat = pysal.open(data_link + '.dbf')
    t1 = time.time()
    tf = t1 - t0
    print 'Loading data:\t\t\t%.5f seconds'%tf
    t0 = time.time()
    yx = [yvar]
    yx.extend(xvars)
    yx = map(nat.header.index, yx)
    yx = [map(row.__getitem__, yx) for row in nat]
    yx = np.array(yx)
    y, x = np.reshape(yx[:, 0], (w.n, 1)), yx[:, 1:]
    t1 = time.time()
    tf = t1 - t0
    print 'Creating y & x vars:\t\t%.5f seconds'%tf
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
    tff=time.time()
    tt = tff - ti
    total = 'Total final time:\t\t%.5f seconds\n'%tt
    print total
    return [ols]

larea = '/home/dani/AAA/LargeData/Seattle/parcel2000_city_resbldg99_larea'
#larea = '/Volumes/GeoDa/Workspace/Julia/Seattle/parcel2000_city_resbldg99_larea'

resarea = '/home/dani/AAA/LargeData/Seattle/all97_Aug08_bis_resarea_tract'
#resarea = '/Volumes/GeoDa/Workspace/Julia/Seattle/all97_Aug08_bis_resarea_tract'

'''
xvars = ['GRADE', 'BEDROOM', 'BASEAREA', 'FULLBATH', 'YEARBLT', 'CONDTION']
yvar = 'LIVEAREA'
# K= 8
model = run_seattle(larea, yvar, xvars)

xvars = ['firplace', 'bricksto', 'bathfull', 'lake1k', 'park250', 'stories2', \
    'viewall', 'nowhite', 'pcty65', 'pctcol', 'age', 'higrade', 'mindist_li', \
    'sqfttotl_r', 'sqftlive_r', 'apex', 'wx_agep', 'wx_conpr', 'wx_sfhom', \
    'wx_duplx']
yvar = 'lnprice'
model = run_seattle(resarea, yvar, xvars, pts=False)

# K= 20
xvars = ['GRADE', 'BEDROOM', 'BASEAREA', 'FULLBATH', 'YEARBLT', 'CONDTION']
yvar = 'LIVEAREA'
model = run_seattle(larea, yvar, xvars, k=20)
'''

# Min. Threshold
xvars = ['GRADE', 'BEDROOM', 'BASEAREA', 'FULLBATH', 'YEARBLT', 'CONDTION']
yvar = 'LIVEAREA'
model = run_seattle(larea, yvar, xvars, k='min')

