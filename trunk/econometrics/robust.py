import numpy as np
import numpy.linalg as la



def gls_dev(x, y, z, u):
    """
    Feasible generalized least squares for the 2SLS model.  This is intended
    to be called from a 2SLS class, but it can be run manually as seen in the
    example below.

    NOTE: no consistency checks
    
    Parameters
    ----------

    x           : array
                  nxk array of independent variables, including endogenous
                  variables (assumed to be aligned with y)
    y           : array
                  nx1 array of dependent variable
    z           : array
                  nxl array of instruments; typically this includes all
                  exogenous variables from x and instruments; (it is different
                  from h seen elsewhere in the library because it can contain
                  a constant)

    Attributes
    ----------

    results     : tuple
                  first element is a kx1 array of estimated coefficients
                  (i.e. betas) and the second is a kxk array for the
                  variance-covariance matrix (i.e [(Z'X)' omega^-1 (Z'X)]^-1 )

    Examples
    --------

    >>> import numpy as np
    >>> import pysal
    >>> import twosls_sp as TwoSLS_sp
    >>> db=pysal.open("examples/columbus.dbf","r")
    >>> y = np.array(db.by_col("CRIME"))
    >>> y = np.reshape(y, (49,1))
    >>> X = []
    >>> X.append(db.by_col("INC"))
    >>> X.append(db.by_col("HOVAL"))
    >>> X = np.array(X).T
    >>> # instrument for HOVAL with DISCBD
    >>> h = []
    >>> h.append(db.by_col("INC"))
    >>> h.append(db.by_col("DISCBD"))
    >>> h = np.array(h).T
    >>> w = pysal.rook_from_shapefile("examples/columbus.shp")
    >>> w.transform = 'r'
    >>> reg = TwoSLS_sp.TwoSLS_Spatial_dev(X, y, w, w_lags=2)
    >>> reg.betas
    array([[ 45.45909249],
           [  0.41929355],
           [ -1.0410089 ],
           [ -0.25953844]])
    >>> z = np.hstack((np.ones(y.shape), reg.h))
    >>> beta_hat, xptxpi = gls_dev(reg.x, reg.y, z, reg.u)
    >>> beta_hat
    array([[ 51.16882977],
           [  0.32904005],
           [ -1.12721019],
           [ -0.28543096]])

    """
    ztx = np.dot(z.T, x)
    v = get_omega(z, u)
    vi = la.inv(v)
    ztxtvi = np.dot(ztx.T, vi)
    ztxtviztx = np.dot(ztxtvi, ztx)
    ztxtviztxi = la.inv(ztxtviztx)    # [(Z'X)' omega^-1 (Z'X)]^-1
    zty = np.dot(z.T, y)
    ztxtvizty = np.dot(ztxtvi, zty)   #  (Z'X)' omega^-1 (Z'y)
    betas = np.dot(ztxtviztxi, ztxtvizty)
    return betas, ztxtviztxi


def get_omega(rhs, u):
    u2 = u**2
    v = u2 * rhs
    return np.dot(rhs.T, v)           # weighting matrix (omega)
    

def _test():
    import doctest
    doctest.testmod()

if __name__ == '__main__':
    _test()



