"""
Script to test the Spatial Diagnostics module on spreg
"""

import numpy as np
import pysal

trunk = '../../../trunk/econometrics/'

# Data Loading
from econometrics.testing_utils import Test_Data 
from econometrics.diagnostics_sp import akTest
from econometrics.twosls_sp import STSLS_dev as STSLS

## 10000 obs
data = Test_Data(100, 4, 'medium', trunk)

# Moran's I

# LM tests

# AK tests
db=pysal.open(trunk + "examples/columbus.dbf","r")
y = np.array(db.by_col("CRIME"))
y = np.reshape(y, (49,1))
X = []
X.append(db.by_col("INC"))
X.append(db.by_col("HOVAL"))
X = np.array(X).T
# instrument for HOVAL with DISCBD
h = []
h.append(db.by_col("INC"))
h.append(db.by_col("DISCBD"))
h = np.array(h).T
w = pysal.rook_from_shapefile(trunk + "examples/columbus.shp")
w.transform = 'r'

iv = STSLS(X, y, w, w_lags=2)
