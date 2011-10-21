import pysal
import numpy as np
from pysal.weights.user import kernelW_from_shapefile
from econometrics.ols import BaseOLS
from econometrics.twosls import BaseTSLS
path = '../../../trunk/econometrics/examples/'
wk = kernelW_from_shapefile(path+'NAT.shp',function='triangular',idVariable='FIPSNO')
db=pysal.open(path+'NAT.dbf',"r")
y = np.array(db.by_col("HR90"))
y = np.reshape(y, (y.shape[0],1))
X = []
X.append(db.by_col("RD90"))
X.append(db.by_col("DV90"))
X = np.array(X).T
ols = BaseOLS(y,X)
print 'std ols:', np.sqrt(ols.vm.diagonal())
wk.transform = 'o'
ols = BaseOLS(y,X, robust='hac', gwk=wk)
print 'std ols HAC:', np.sqrt(ols.vm.diagonal())
yd = [db.by_col("UE90")]
yd = np.array(yd).T
q = [db.by_col("UE80")]
q = np.array(q).T
tsls = BaseTSLS(y, X, yd, q=q)
print 'std tsls:', np.sqrt(tsls.vm.diagonal())
tsls = BaseTSLS(y, X, yd, q=q, robust='hac', gwk=wk)
print  'std tsls HAC:', np.sqrt(tsls.vm.diagonal())
