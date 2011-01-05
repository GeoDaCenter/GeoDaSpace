import numpy as np
import numpy.linalg as la



def gls4tsls(y, z, h, u):
    """
    Feasible generalized least squares for the 2SLS model.  This is intended
    to be called from a 2SLS class, but it can be run manually as seen in the
    example below.

    NOTE: no consistency checks
    
    Parameters
    ----------

    y           : array
                  nx1 array of dependent variable
    z           : array
                  nxk array of independent variables, including endogenous
                  variables (assumed to be aligned with y)
    h           : array
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
    >>> from twosls_sp import BaseSTSLS
    >>> w = pysal.rook_from_shapefile("examples/columbus.shp")
    >>> w.transform = 'r'
    >>> db=pysal.open("examples/columbus.dbf","r")
    >>> y = np.array(db.by_col("CRIME"))
    >>> y = np.reshape(y, (49,1))
    >>> # no non-spatial endogenous variables
    >>> X = []
    >>> X.append(db.by_col("INC"))
    >>> X.append(db.by_col("HOVAL"))
    >>> X = np.array(X).T
    >>> # run gls4tsls manually
    >>> reg=BaseSTSLS(y, X, w, w_lags=2)
    >>> reg.betas
    array([[ 45.45909249],
           [ -1.0410089 ],
           [ -0.25953844],
           [  0.41929355]])
    >>> beta_hat, xptxpi = gls4tsls(reg.y, reg.z, reg.h, reg.u)
    >>> beta_hat
    array([[ 51.16882977],
           [ -1.12721019],
           [ -0.28543096],
           [  0.32904005]])
    >>> # run gls4tsls directly using 2SLS
    >>> reg=BaseSTSLS(y, X, w, w_lags=2, robust='gls')
    >>> reg.betas
    array([[ 51.16882977],
           [ -1.12721019],
           [ -0.28543096],
           [  0.32904005]])

    """
    htz = np.dot(h.T, z)
    v = get_omega(h, u)
    vi = la.inv(v)
    htztvi = np.dot(htz.T, vi)
    htztvihtz = np.dot(htztvi, htz)
    htztvihtzi = la.inv(htztvihtz)    # [(h'Z)' omega^-1 (h'Z)]^-1
    hty = np.dot(h.T, y)
    htztvihty = np.dot(htztvi, hty)   #  (h'Z)' omega^-1 (h'y)
    betas = np.dot(htztvihtzi, htztvihty)
    return betas, htztvihtzi


def get_omega(rhs, u):
    u2 = u**2
    v = u2 * rhs
    return np.dot(rhs.T, v)           # weighting matrix (omega)
    

def _test():
    import doctest
    doctest.testmod()

if __name__ == '__main__':
    _test()



