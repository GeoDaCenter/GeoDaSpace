


from pysal.spreg.ols import OLS
from twosls import TSLS
from twosls_sp import STSLS
from spHetErr import SWLS_Het, GSTSLS_Het, GSTSLS_Het_lag
from spError import GSTSLS, GMSWLS, GSTSLS_lag

def spmodel(name_ds, w, y, name_y, x, name_x, ye, name_ye,\
                h, name_h, r, name_r, s, name_s, t, name_t,\
                model_type, endog, spat_tests, std_err):
    """
    A single function to call all the econometric models in pysal.

    Parameters
    ----------

    name_ds     : string
                  Name of dataset for use in output
    w           : spatial weights object
                  PySAL W object
    y           : array
                  nx1 array of dependent variable
    name_y      : string
                  Name of dependent variable for use in output
    x           : array
                  array of independent variables, excluding endogenous
                  variables (two dimensional array)
    name_x      : list of strings
                  Names of independent variables for use in output
    ye          : array
                  array of endogenous variables (two dimensional array)
    name_ye     : list of strings
                  Names of endogenous variables for use in output
    h           : array
                  array of external exogenous variables to use as instruments
                  (note: this should not contain any variables from x)
                  (two dimensional array)
    name_h      : list of strings
                  Names of instruments for use in output
    r           : place holder
                  regimes place holder
    name_r      : place holder
                  regimes place holder
    t           : place holder
                  time place holder
    name_t      : place holder
                  time place holder
    s           : place holder
                  cross section place holder
    name_s      : place holder
                  cross section place holder
    model_type  : string
                  Options: 'Standard', 'Spatial Lag', 'Spatial Error', 
                  'Spatial Lag+Error'; must be one of these options
    spat_tests  : boolean
                  Boolean indicating whether spatial tests should be run
    std_err     : string
                  Options: 'White', 'HAC', 'KP HAC', ''; the empty string will    LA: Note, should be KP HET?
                  return unadjusted standard errors

    Returns
    -------
    return  : PySAL regression object

    Examples
    --------

    Not all combinations are tested below

    >>> import numpy as np
    >>> import pysal
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
    
    No non-spatial endogenous variables
    
    >>> reg = spmodel(name_ds='columbus', w=w, y=y, name_y='crime', x=X, name_x=['inc', 'hoval'],\
        ye=[], name_ye=[], h=[], name_h=[],\
        r=None, name_r=None, s=None, name_s=None, t=None, name_t=None,\
        model_type='Standard', endog=False, spat_tests=False, std_err='')
    >>> print reg.name_x
    ['CONSTANT', 'inc', 'hoval']
    >>> reg = spmodel(name_ds='columbus', w=w, y=y, name_y='crime', x=X, name_x=['inc', 'hoval'],\
        ye=[], name_ye=[], h=[], name_h=[],\
        r=None, name_r=None, s=None, name_s=None, t=None, name_t=None,\
        model_type='Spatial Lag', endog=False, spat_tests=False, std_err='')
    >>> print reg.name_z
    ['CONSTANT', 'inc', 'hoval', 'lag_crime']
    >>> reg = spmodel(name_ds='columbus', w=w, y=y, name_y='crime', x=X, name_x=['inc', 'hoval'],\
        ye=[], name_ye=[], h=[], name_h=[],\
        r=None, name_r=None, s=None, name_s=None, t=None, name_t=None,\
        model_type='Spatial Error', endog=False, spat_tests=False, std_err='')
    >>> print reg.name_x
    ['CONSTANT', 'inc', 'hoval', 'lambda']
    >>> reg = spmodel(name_ds='columbus', w=w, y=y, name_y='crime', x=X, name_x=['inc', 'hoval'],\
        ye=[], name_ye=[], h=[], name_h=[],\
        r=None, name_r=None, s=None, name_s=None, t=None, name_t=None,\
        model_type='Spatial Lag+Error', endog=False, spat_tests=False, std_err='')
    >>> print reg.name_z
    ['CONSTANT', 'inc', 'hoval', 'lag_crime', 'lambda']
    >>> reg = spmodel(name_ds='columbus', w=w, y=y, name_y='crime', x=X, name_x=['inc', 'hoval'],\
        ye=[], name_ye=[], h=[], name_h=[],\
        r=None, name_r=None, s=None, name_s=None, t=None, name_t=None,\
        model_type='Spatial Lag+Error', endog=False, spat_tests=False, std_err='KP HET')
    >>> print reg.name_z
    ['CONSTANT', 'inc', 'hoval', 'lag_crime', 'lambda']
    
    Add in non-spatial endogenous variables

    >>> X = np.array(db.by_col("INC"))
    >>> X = np.reshape(X, (49,1))
    >>> yd = np.array(db.by_col("HOVAL"))
    >>> yd = np.reshape(yd, (49,1))
    >>> q = np.array(db.by_col("DISCBD"))
    >>> q = np.reshape(q, (49,1))
    >>> reg = spmodel(name_ds='columbus', w=w, y=y, name_y='crime', x=X, name_x=['inc'],\
        ye=yd, name_ye=['hoval'], h=q, name_h=['discbd'],\
        r=None, name_r=None, s=None, name_s=None, t=None, name_t=None,\
        model_type='Standard', endog=True, spat_tests=False, std_err='')
    >>> print reg.name_z
    ['CONSTANT', 'inc', 'hoval']
    >>> reg = spmodel(name_ds='columbus', w=w, y=y, name_y='crime', x=X, name_x=['inc'],\
        ye=yd, name_ye=['hoval'], h=q, name_h=['discbd'],\
        r=None, name_r=None, s=None, name_s=None, t=None, name_t=None,\
        model_type='Spatial Lag', endog=True, spat_tests=False, std_err='')
    >>> print reg.name_z
    ['CONSTANT', 'inc', 'hoval', 'lag_crime']
    >>> reg = spmodel(name_ds='columbus', w=w, y=y, name_y='crime', x=X, name_x=['inc'],\
        ye=yd, name_ye=['hoval'], h=q, name_h=['discbd'],\
        r=None, name_r=None, s=None, name_s=None, t=None, name_t=None,\
        model_type='Spatial Lag', endog=True, spat_tests=False, std_err='White')
    >>> print reg.name_z
    ['CONSTANT', 'inc', 'hoval', 'lag_crime']
    >>> reg = spmodel(name_ds='columbus', w=w, y=y, name_y='crime', x=X, name_x=['inc'],\
        ye=yd, name_ye=['hoval'], h=q, name_h=['discbd'],\
        r=None, name_r=None, s=None, name_s=None, t=None, name_t=None,\
        model_type='Spatial Error', endog=True, spat_tests=False, std_err='')
    >>> print reg.name_z
    ['CONSTANT', 'inc', 'hoval', 'lambda']
    >>> reg = spmodel(name_ds='columbus', w=w, y=y, name_y='crime', x=X, name_x=['inc'],\
        ye=yd, name_ye=['hoval'], h=q, name_h=['discbd'],\
        r=None, name_r=None, s=None, name_s=None, t=None, name_t=None,\
        model_type='Spatial Lag+Error', endog=True, spat_tests=False, std_err='')
    >>> print reg.name_z
    ['CONSTANT', 'inc', 'hoval', 'lag_crime', 'lambda']
    >>> reg = spmodel(name_ds='columbus', w=w, y=y, name_y='crime', x=X, name_x=['inc'],\
        ye=yd, name_ye=['hoval'], h=q, name_h=['discbd'],\
        r=None, name_r=None, s=None, name_s=None, t=None, name_t=None,\
        model_type='Spatial Lag+Error', endog=True, spat_tests=False, std_err='KP HET')
    >>> print reg.name_z
    ['CONSTANT', 'inc', 'hoval', 'lag_crime', 'lambda']

    """
    if model_type == 'Standard':
        if endog:
            if spat_tests:
                if std_err == 'White':
                    # 2SLS with White std errors and with spatial diagnostics
                    return TSLS(y=y, x=x, yend=ye, q=h, w=w, robust='white',\
                                name_y=name_y, name_x=name_x, name_yend=name_ye,\
                                name_q=name_h, name_ds=name_ds)
                elif std_err == 'HAC':  # LA standard 2SLS needs HAC
                    raise Exception, "not yet implemented"
                elif std_err == '':
                    # 2SLS with spatial diagnostics
                    return TSLS(y=y, x=x, yend=ye, q=h, w=w,\
                                name_y=name_y, name_x=name_x, name_yend=name_ye,\
                                name_q=name_h, name_ds=name_ds)
            else:
                if std_err == 'White':
                    # 2SLS with White std errors
                    return TSLS(y=y, x=x, yend=ye, q=h, robust='White',\
                                name_y=name_y, name_x=name_x, name_yend=name_ye,\
                                name_q=name_h, name_ds=name_ds)
                elif std_err == 'HAC':  # LA standard 2SLS needs HAC
                    raise Exception, "not yet implemented"
                elif std_err == '':
                    # 2SLS
                    return TSLS(y=y, x=x, yend=ye, q=h,\
                                name_y=name_y, name_x=name_x, name_yend=name_ye,\
                                name_q=name_h, name_ds=name_ds)
        else:   # LA NOTE: OLS needs White and HAC
            if spat_tests:
                # OLS with spatial diagnostics
                return OLS(y=y, x=x, w=w,\
                           name_y=name_y, name_x=name_x, name_ds=name_ds)
            else:
                # OLS without spatial diagnostics
                return OLS(y=y, x=x,\
                           name_y=name_y, name_x=name_x, name_ds=name_ds)

    elif model_type == 'Spatial Lag':
        if std_err == 'KP HET':
            raise Exception, "not a valid combination"
        elif std_err == 'White':
            if endog:
               #GM Spatial lag with White std. errors with non-spatial endog variables
                return STSLS(y=y, x=x, w=w, yend=ye, q=h,\
                             name_y=name_y, name_x=name_x, name_yend=name_ye,\
                             name_q=name_h, name_ds=name_ds,\
                             robust='white')
            else:
               #GM Spatial lag with White std. errors
                return STSLS(y=y, x=x, w=w,\
                             name_y=name_y, name_x=name_x, name_ds=name_ds,\
                             robust='white')
        elif std_err == 'HAC':
            raise Exception, "not yet implemented"  # LA: HAC is valid with spatial lag
        elif std_err == '':
            if endog:
               #GM Spatial lag with non-spatial endog variables
                return STSLS(y=y, x=x, w=w, yend=ye, q=h,\
                             name_y=name_y, name_x=name_x, name_yend=name_ye,\
                             name_q=name_h, name_ds=name_ds)
            else:
               #GM Spatial lag
                return STSLS(y=y, x=x, w=w,\
                             name_y=name_y, name_x=name_x, name_ds=name_ds)
        else:
            raise Exception, "invalid option passed to std_err"

    elif model_type == 'Spatial Error':
        if std_err == 'KP HET':
            if endog:
                #GM Spatial error with het and with non-spatial endogenous variable
                return GSTSLS_Het(y=y, x=x, w=w, yend=ye, q=h,\
                            name_y=name_y, name_x=name_x, name_yend=name_ye,\
                            name_q=name_h, name_ds=name_ds)
            else:
                #GM Spatial error with het
                return SWLS_Het(y=y, x=x, w=w,\
                             name_y=name_y, name_x=name_x, name_ds=name_ds)
        elif std_err == 'White':
            raise Exception, "not a valid combination"
        elif std_err == 'HAC':    # NOTE LA: HAC should not be for spatial error
            raise Exception, "not a valid combination"
        elif std_err == '':
            if endog:
                #GM Spatial error with non-spatial endogenous variable
                return GSTSLS(y=y, x=x, w=w, yend=ye, q=h,\
                              name_y=name_y, name_x=name_x, name_yend=name_ye,\
                              name_q=name_h, name_ds=name_ds)
            else:
                #GM Spatial error
                return GMSWLS(y=y, x=x, w=w,\
                           name_y=name_y, name_x=name_x, name_ds=name_ds)
        else:
            raise Exception, "invalid option passed to std_err"

    elif model_type == 'Spatial Lag+Error':
        if std_err == 'KP HET':  
            if endog:
                #GM Spatial combo with het with non-spatial endogenous variables
                return GSTSLS_Het_lag(y=y, x=x, w=w, yend=ye, q=h,\
                             name_y=name_y, name_x=name_x, name_yend=name_ye,\
                             name_q=name_h, name_ds=name_ds)
            else:
                #GM Spatial combo with het
                return GSTSLS_Het_lag(y=y, x=x, w=w,\
                             name_y=name_y, name_x=name_x, name_ds=name_ds)
        elif std_err == 'White':
            raise Exception, "not a valid combination"  # no white with lag-error model, subsumed in KP-HET
        elif std_err == 'HAC':
            raise Exception, "not a valid combination"  # HAC should be in spatial lag, not here
        elif std_err == '':
            if endog:
                #GM Spatial combo with non-spatial endogenous variables
                return GSTSLS_lag(y=y, x=x, w=w, yend=ye, q=h,\
                              name_y=name_y, name_x=name_x, name_yend=name_ye,\
                              name_q=name_h, name_ds=name_ds)
            else:
                #GM Spatial combo
                return GSTSLS_lag(y=y, x=x, w=w,\
                           name_y=name_y, name_x=name_x, name_ds=name_ds)
        else:
            raise Exception, "invalid option passed to std_err"
    
        
def _test():
    import doctest
    doctest.testmod()

if __name__ == '__main__':
    _test()




