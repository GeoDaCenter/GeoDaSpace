import copy as COPY
from ols import OLS
from twosls import TSLS
from twosls_sp import GM_Lag
from spHetErr import GM_Error_Het, GM_Endog_Error_Het, GM_Combo_Het
from spError import GM_Endog_Error, GM_Error, GM_Combo
from spError import GM_Endog_Error_Hom, GM_Error_Hom, GM_Combo_Hom
import robust as ROBUST
import user_output as USER

def spmodel(name_ds, w_list, wk_list, y, name_y, x, name_x, ye, name_ye,\
                h, name_h, r, name_r, s, name_s, t, name_t,\
                model_type, endog, nonspat_diag, spat_diag,\
                sig2n_k_ols, sig2n_k_tsls, sig2n_k_gmlag,\
                white, hac, kp_het, gm):
    """
    A single function to call all the econometric models in pysal.

    Parameters
    ----------

    name_ds     : string
                  Name of dataset for use in output
    w_list      : list
                  List of PySAL W objects
    wk_list     : list
                  List of PySAL W kernel objects
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
    endog       : boolean
                  If True, use the ye and h arrays in the regession (i.e., a
                  2SLS type approach); if False ignore ye and h
    nonspat_diag : boolean
                   Run nonspatial diagnositcs (this might be removed later or
                   only applied to OLS)
    spat_diag   : boolean
                  Run spatial tests such as Moran's I on the residuals, LM and
                  AK tests
    sig2n_k_ols : boolean
                  If Ture use n-k to compute OLS standard errors; if False use n
    sig2n_k_tsls : boolean
                   If Ture use N-k to compute 2SLS standard errors; if False use n
    sig2n_k_gmlag : boolean
                    If Ture use N-k to compute GM Lag standard errors; if False use n
    white       : boolean
                  Compute White standard errors
    hac         : boolean
                  Compute HAC standard errors
    kp_het      : boolean
                  Run regression using KP2010 GMM approach for heteroskedasticity

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
    >>> wk = pysal.kernelW_from_shapefile("examples/columbus.shp",k=4,function='epanechnikov',idVariable=None,fixed=False)
    >>> db=pysal.open("examples/columbus.dbf","r")
    >>> y = np.array(db.by_col("CRIME"))
    >>> y = np.reshape(y, (49,1))
    >>> # no non-spatial endogenous variables
    >>> X = []
    >>> X.append(db.by_col("INC"))
    >>> X.append(db.by_col("HOVAL"))
    >>> X = np.array(X).T
    
    No non-spatial endogenous variables
    
    >>> reg = spmodel(name_ds='columbus', w_list=[w], wk_list=[], y=y, name_y='crime', x=X, name_x=['inc', 'hoval'],\
        ye=[], name_ye=[], h=[], name_h=[],\
        r=None, name_r=None, s=None, name_s=None, t=None, name_t=None,\
        model_type='Standard', endog=False, nonspat_diag=True, spat_diag=True,\
        sig2n_k_ols=True, sig2n_k_tsls=False, sig2n_k_gmlag=False,\
        white=False, hac=False, kp_het=False, gm=False)
    >>> print reg[0].name_x
    ['CONSTANT', 'inc', 'hoval']
    >>> reg = spmodel(name_ds='columbus', w_list=[w], wk_list=[], y=y, name_y='crime', x=X, name_x=['inc', 'hoval'],\
        ye=[], name_ye=[], h=[], name_h=[],\
        r=None, name_r=None, s=None, name_s=None, t=None, name_t=None,\
        model_type='Standard', endog=False, nonspat_diag=True, spat_diag=True,\
        sig2n_k_ols=True, sig2n_k_tsls=False, sig2n_k_gmlag=False,\
        white=True, hac=False, kp_het=False, gm=False)
    >>> print reg[0].name_x
    ['CONSTANT', 'inc', 'hoval']
    >>> reg = spmodel(name_ds='columbus', w_list=[w], wk_list=[wk], y=y, name_y='crime', x=X, name_x=['inc', 'hoval'],\
        ye=[], name_ye=[], h=[], name_h=[],\
        r=None, name_r=None, s=None, name_s=None, t=None, name_t=None,\
        model_type='Standard', endog=False, nonspat_diag=True, spat_diag=False,\
        sig2n_k_ols=True, sig2n_k_tsls=False, sig2n_k_gmlag=False,\
        white=False, hac=True, kp_het=False, gm=False)
    >>> print reg[0].name_x
    ['CONSTANT', 'inc', 'hoval']
    >>> reg = spmodel(name_ds='columbus', w_list=[w], wk_list=[], y=y, name_y='crime', x=X, name_x=['inc', 'hoval'],\
        ye=[], name_ye=[], h=[], name_h=[],\
        r=None, name_r=None, s=None, name_s=None, t=None, name_t=None,\
        model_type='Spatial Lag', endog=False, nonspat_diag=True, spat_diag=False,\
        sig2n_k_ols=True, sig2n_k_tsls=False, sig2n_k_gmlag=False,\
        white=False, hac=False, kp_het=False, gm=False)
    >>> print reg[0].name_z
    ['CONSTANT', 'inc', 'hoval', 'W_crime']
    >>> reg = spmodel(name_ds='columbus', w_list=[w], wk_list=[], y=y, name_y='crime', x=X, name_x=['inc', 'hoval'],\
        ye=[], name_ye=[], h=[], name_h=[],\
        r=None, name_r=None, s=None, name_s=None, t=None, name_t=None,\
        model_type='Spatial Lag', endog=False, nonspat_diag=True, spat_diag=True,\
        sig2n_k_ols=True, sig2n_k_tsls=False, sig2n_k_gmlag=False,\
        white=True, hac=False, kp_het=False, gm=False)
    >>> print reg[0].name_z
    ['CONSTANT', 'inc', 'hoval', 'W_crime']
    >>> reg = spmodel(name_ds='columbus', w_list=[w], wk_list=[], y=y, name_y='crime', x=X, name_x=['inc', 'hoval'],\
        ye=[], name_ye=[], h=[], name_h=[],\
        r=None, name_r=None, s=None, name_s=None, t=None, name_t=None,\
        model_type='Spatial Lag', endog=False, nonspat_diag=True, spat_diag=False,\
        sig2n_k_ols=True, sig2n_k_tsls=False, sig2n_k_gmlag=False,\
        white=False, hac=True, kp_het=False, gm=False)
    >>> print reg[0].name_z
    ['CONSTANT', 'inc', 'hoval', 'W_crime']
    >>> reg = spmodel(name_ds='columbus', w_list=[w], wk_list=[], y=y, name_y='crime', x=X, name_x=['inc', 'hoval'],\
        ye=[], name_ye=[], h=[], name_h=[],\
        r=None, name_r=None, s=None, name_s=None, t=None, name_t=None,\
        model_type='Spatial Error', endog=False, nonspat_diag=True, spat_diag=False,\
        sig2n_k_ols=True, sig2n_k_tsls=False, sig2n_k_gmlag=False,\
        white=False, hac=False, kp_het=False, gm=False)
    >>> print reg[0].name_x
    ['CONSTANT', 'inc', 'hoval', 'lambda']
    >>> reg = spmodel(name_ds='columbus', w_list=[w], wk_list=[], y=y, name_y='crime', x=X, name_x=['inc', 'hoval'],\
        ye=[], name_ye=[], h=[], name_h=[],\
        r=None, name_r=None, s=None, name_s=None, t=None, name_t=None,\
        model_type='Spatial Error', endog=False, nonspat_diag=True, spat_diag=True,\
        sig2n_k_ols=True, sig2n_k_tsls=False, sig2n_k_gmlag=False,\
        white=False, hac=False, kp_het=True, gm=False)
    >>> print reg[0].name_x
    ['CONSTANT', 'inc', 'hoval', 'lambda']
    >>> reg = spmodel(name_ds='columbus', w_list=[w], wk_list=[], y=y, name_y='crime', x=X, name_x=['inc', 'hoval'],\
        ye=[], name_ye=[], h=[], name_h=[],\
        r=None, name_r=None, s=None, name_s=None, t=None, name_t=None,\
        model_type='Spatial Error', endog=False, nonspat_diag=True, spat_diag=False,\
        sig2n_k_ols=True, sig2n_k_tsls=False, sig2n_k_gmlag=False,\
        white=False, hac=False, kp_het=False, gm=True)
    >>> print reg[0].name_x
    ['CONSTANT', 'inc', 'hoval', 'lambda']
    >>> reg = spmodel(name_ds='columbus', w_list=[w], wk_list=[], y=y, name_y='crime', x=X, name_x=['inc', 'hoval'],\
        ye=[], name_ye=[], h=[], name_h=[],\
        r=None, name_r=None, s=None, name_s=None, t=None, name_t=None,\
        model_type='Spatial Error', endog=False, nonspat_diag=True, spat_diag=False,\
        sig2n_k_ols=True, sig2n_k_tsls=False, sig2n_k_gmlag=False,\
        white=False, hac=False, kp_het=True, gm=True)
    >>> print reg[0].name_x
    ['CONSTANT', 'inc', 'hoval', 'lambda']
    >>> reg = spmodel(name_ds='columbus', w_list=[w], wk_list=[], y=y, name_y='crime', x=X, name_x=['inc', 'hoval'],\
        ye=[], name_ye=[], h=[], name_h=[],\
        r=None, name_r=None, s=None, name_s=None, t=None, name_t=None,\
        model_type='Spatial Lag+Error', endog=False, nonspat_diag=True, spat_diag=False,\
        sig2n_k_ols=True, sig2n_k_tsls=False, sig2n_k_gmlag=False,\
        white=False, hac=False, kp_het=False, gm=False)
    >>> print reg[0].name_z
    ['CONSTANT', 'inc', 'hoval', 'W_crime', 'lambda']
    >>> reg = spmodel(name_ds='columbus', w_list=[w], wk_list=[], y=y, name_y='crime', x=X, name_x=['inc', 'hoval'],\
        ye=[], name_ye=[], h=[], name_h=[],\
        r=None, name_r=None, s=None, name_s=None, t=None, name_t=None,\
        model_type='Spatial Lag+Error', endog=False, nonspat_diag=True, spat_diag=False,\
        sig2n_k_ols=True, sig2n_k_tsls=False, sig2n_k_gmlag=False,\
        white=False, hac=False, kp_het=True, gm=False)
    >>> print reg[0].name_z
    ['CONSTANT', 'inc', 'hoval', 'W_crime', 'lambda']
    
    Add in non-spatial endogenous variables

    >>> X = np.array(db.by_col("INC"))
    >>> X = np.reshape(X, (49,1))
    >>> yd = np.array(db.by_col("HOVAL"))
    >>> yd = np.reshape(yd, (49,1))
    >>> q = np.array(db.by_col("DISCBD"))
    >>> q = np.reshape(q, (49,1))
    >>> reg = spmodel(name_ds='columbus', w_list=[w], wk_list=[], y=y, name_y='crime', x=X, name_x=['inc'],\
        ye=yd, name_ye=['hoval'], h=q, name_h=['discbd'],\
        r=None, name_r=None, s=None, name_s=None, t=None, name_t=None,\
        model_type='Standard', endog=True, nonspat_diag=True, spat_diag=False,\
        sig2n_k_ols=True, sig2n_k_tsls=False, sig2n_k_gmlag=False,\
        white=False, hac=False, kp_het=False, gm=False)
    >>> print reg[0].name_z
    ['CONSTANT', 'inc', 'hoval']
    >>> reg = spmodel(name_ds='columbus', w_list=[w], wk_list=[], y=y, name_y='crime', x=X, name_x=['inc'],\
        ye=yd, name_ye=['hoval'], h=q, name_h=['discbd'],\
        r=None, name_r=None, s=None, name_s=None, t=None, name_t=None,\
        model_type='Standard', endog=True, nonspat_diag=True, spat_diag=False,\
        sig2n_k_ols=True, sig2n_k_tsls=False, sig2n_k_gmlag=False,\
        white=True, hac=False, kp_het=False, gm=False)
    >>> print reg[0].name_z
    ['CONSTANT', 'inc', 'hoval']
    >>> reg = spmodel(name_ds='columbus', w_list=[w], wk_list=[wk], y=y, name_y='crime', x=X, name_x=['inc'],\
        ye=yd, name_ye=['hoval'], h=q, name_h=['discbd'],\
        r=None, name_r=None, s=None, name_s=None, t=None, name_t=None,\
        model_type='Standard', endog=True, nonspat_diag=True, spat_diag=True,\
        sig2n_k_ols=True, sig2n_k_tsls=False, sig2n_k_gmlag=False,\
        white=False, hac=True, kp_het=False, gm=False)
    >>> print reg[0].name_z
    ['CONSTANT', 'inc', 'hoval']
    >>> reg = spmodel(name_ds='columbus', w_list=[w], wk_list=[], y=y, name_y='crime', x=X, name_x=['inc'],\
        ye=yd, name_ye=['hoval'], h=q, name_h=['discbd'],\
        r=None, name_r=None, s=None, name_s=None, t=None, name_t=None,\
        model_type='Spatial Lag', endog=True, nonspat_diag=True, spat_diag=False,\
        sig2n_k_ols=True, sig2n_k_tsls=False, sig2n_k_gmlag=False,\
        white=False, hac=False, kp_het=False, gm=False)
    >>> print reg[0].name_z
    ['CONSTANT', 'inc', 'hoval', 'W_crime']
    >>> reg = spmodel(name_ds='columbus', w_list=[w], wk_list=[], y=y, name_y='crime', x=X, name_x=['inc'],\
        ye=yd, name_ye=['hoval'], h=q, name_h=['discbd'],\
        r=None, name_r=None, s=None, name_s=None, t=None, name_t=None,\
        model_type='Spatial Lag', endog=True, nonspat_diag=True, spat_diag=False,\
        sig2n_k_ols=True, sig2n_k_tsls=False, sig2n_k_gmlag=False,\
        white=True, hac=False, kp_het=False, gm=False)
    >>> print reg[0].name_z
    ['CONSTANT', 'inc', 'hoval', 'W_crime']
    >>> reg = spmodel(name_ds='columbus', w_list=[w], wk_list=[], y=y, name_y='crime', x=X, name_x=['inc'],\
        ye=yd, name_ye=['hoval'], h=q, name_h=['discbd'],\
        r=None, name_r=None, s=None, name_s=None, t=None, name_t=None,\
        model_type='Spatial Lag', endog=True, nonspat_diag=True, spat_diag=False,\
        sig2n_k_ols=True, sig2n_k_tsls=False, sig2n_k_gmlag=False,\
        white=False, hac=True, kp_het=False, gm=False)
    >>> print reg[0].name_z
    ['CONSTANT', 'inc', 'hoval', 'W_crime']
    >>> reg = spmodel(name_ds='columbus', w_list=[w], wk_list=[], y=y, name_y='crime', x=X, name_x=['inc'],\
        ye=yd, name_ye=['hoval'], h=q, name_h=['discbd'],\
        r=None, name_r=None, s=None, name_s=None, t=None, name_t=None,\
        model_type='Spatial Error', endog=True, nonspat_diag=True, spat_diag=True,\
        sig2n_k_ols=True, sig2n_k_tsls=False, sig2n_k_gmlag=False,\
        white=False, hac=False, kp_het=False, gm=False)
    >>> print reg[0].name_z
    ['CONSTANT', 'inc', 'hoval', 'lambda']
    >>> reg = spmodel(name_ds='columbus', w_list=[w], wk_list=[], y=y, name_y='crime', x=X, name_x=['inc'],\
        ye=yd, name_ye=['hoval'], h=q, name_h=['discbd'],\
        r=None, name_r=None, s=None, name_s=None, t=None, name_t=None,\
        model_type='Spatial Error', endog=True, nonspat_diag=True, spat_diag=True,\
        sig2n_k_ols=True, sig2n_k_tsls=False, sig2n_k_gmlag=False,\
        white=False, hac=False, kp_het=True, gm=False)
    >>> print reg[0].name_z
    ['CONSTANT', 'inc', 'hoval', 'lambda']
    >>> reg = spmodel(name_ds='columbus', w_list=[w], wk_list=[], y=y, name_y='crime', x=X, name_x=['inc'],\
        ye=yd, name_ye=['hoval'], h=q, name_h=['discbd'],\
        r=None, name_r=None, s=None, name_s=None, t=None, name_t=None,\
        model_type='Spatial Lag+Error', endog=True, nonspat_diag=True, spat_diag=False,\
        sig2n_k_ols=True, sig2n_k_tsls=False, sig2n_k_gmlag=False,\
        white=False, hac=False, kp_het=False, gm=False)
    >>> print reg[0].name_z
    ['CONSTANT', 'inc', 'hoval', 'W_crime', 'lambda']
    >>> reg = spmodel(name_ds='columbus', w_list=[w], wk_list=[], y=y, name_y='crime', x=X, name_x=['inc'],\
        ye=yd, name_ye=['hoval'], h=q, name_h=['discbd'],\
        r=None, name_r=None, s=None, name_s=None, t=None, name_t=None,\
        model_type='Spatial Lag+Error', endog=True, nonspat_diag=True, spat_diag=False,\
        sig2n_k_ols=True, sig2n_k_tsls=False, sig2n_k_gmlag=False,\
        white=False, hac=False, kp_het=True, gm=False)
    >>> print reg[0].name_z
    ['CONSTANT', 'inc', 'hoval', 'W_crime', 'lambda']

    """
    output = []
    if model_type == 'Standard':
        if endog:
            # run 2SLS regression; with or without nonspatial diagnostics
            reg = TSLS(y=y, x=x, yend=ye, q=h,\
                        nonspat_diag=nonspat_diag, spat_diag=False,\
                        name_y=name_y, name_x=name_x, name_yend=name_ye,\
                        name_q=name_h, name_ds=name_ds, sig2n_k=sig2n_k_tsls)
        else:
            # run OLS regression; with or without nonspatial diagnostics
            reg = OLS(y=y, x=x,\
                       nonspat_diag=nonspat_diag, spat_diag=False,\
                       name_y=name_y, name_x=name_x, name_ds=name_ds,
                       sig2n_k=sig2n_k_ols)
        if w_list and spat_diag:
            for w in w_list:
                # add spatial diagnostics for each W
                reg_spat = COPY.copy(reg)
                reg_spat._get_diagnostics(w=w, beta_diag=False,\
                                          nonspat_diag=False, spat_diag=True)
                output.append(reg_spat)
        else:
            output.append(reg)
        if white:
            # compute White std errors
            output.append(get_robust(reg, 'white'))
        if hac:
            for wk in wk_list:
                # compute HAC std errors
                output.append(get_robust(reg, 'hac', wk))
        return output



    elif model_type == 'Spatial Lag':
        for w in w_list:
            if endog:
               #GM Spatial lag with non-spatial endog variables
                output.append(GM_Lag(y=y, x=x, w=w, yend=ye, q=h,\
                             name_y=name_y, name_x=name_x, name_yend=name_ye,\
                             name_q=name_h, name_ds=name_ds, sig2n_k=sig2n_k_gmlag))
            else:
               #GM Spatial lag
                output.append(GM_Lag(y=y, x=x, w=w,\
                             name_y=name_y, name_x=name_x, name_ds=name_ds,\
                             sig2n_k=sig2n_k_gmlag))
        white_regs = []
        if white:
            for reg in output:
                # compute White std errors
                white_regs.append(get_robust(reg, 'white'))
        hac_regs = []
        if hac:
            for reg in output:
                for wk in wk_list:
                    # compute HAC std errors
                    hac_regs.append(get_robust(reg, 'hac', wk))
        output.extend(white_regs)
        output.extend(hac_regs)
        return output



    elif model_type == 'Spatial Error':
        kp_het_regs = []
        if gm:
            for w in w_list:
                if endog:
                    #GM Spatial error (hom) with non-spatial endogenous variable
                    output.append(GM_Endog_Error_Hom(y=y, x=x, w=w, yend=ye, q=h,\
                                  name_y=name_y, name_x=name_x, name_yend=name_ye,\
                                  name_q=name_h, name_ds=name_ds))
                else:
                    #GM Spatial error (hom) 
                    output.append(GM_Error_Hom(y=y, x=x, w=w,\
                                  name_y=name_y, name_x=name_x, name_ds=name_ds))
        else:
            for w in w_list:
                if endog:
                    #GM Spatial error with non-spatial endogenous variable (KP 98-99)
                    output.append(GM_Endog_Error(y=y, x=x, w=w, yend=ye, q=h,\
                                  name_y=name_y, name_x=name_x, name_yend=name_ye,\
                                  name_q=name_h, name_ds=name_ds))
                else:
                    #GM Spatial error (KP 98-99)
                    output.append(GM_Error(y=y, x=x, w=w,\
                                  name_y=name_y, name_x=name_x, name_ds=name_ds))
        if kp_het:
            for w in w_list:
                if endog:
                    #GM Spatial error with het and with non-spatial endogenous variable
                    kp_het_regs.append(GM_Endog_Error_Het(y=y, x=x, w=w, yend=ye, q=h,\
                                        name_y=name_y, name_x=name_x, name_yend=name_ye,\
                                        name_q=name_h, name_ds=name_ds))
                else:
                    #GM Spatial error with het
                    kp_het_regs.append(GM_Error_Het(y=y, x=x, w=w,\
                                         name_y=name_y, name_x=name_x, name_ds=name_ds))
        output.extend(kp_het_regs)
        return output



    elif model_type == 'Spatial Lag+Error':
        kp_het_regs = []
        if gm:
            for w in w_list:
                if endog:
                    #GM Spatial combo (hom) with non-spatial endogenous variables
                    output.append(GM_Combo_Hom(y=y, x=x, w=w, yend=ye, q=h,\
                                  name_y=name_y, name_x=name_x, name_yend=name_ye,\
                                  name_q=name_h, name_ds=name_ds))
                else:
                    #GM Spatial combo (hom)
                    output.append(GM_Combo_Hom(y=y, x=x, w=w,\
                                  name_y=name_y, name_x=name_x, name_ds=name_ds))
        else:
            for w in w_list:
                if endog:
                    #GM Spatial combo with non-spatial endogenous variables (KP 98-99)
                    output.append(GM_Combo(y=y, x=x, w=w, yend=ye, q=h,\
                                  name_y=name_y, name_x=name_x, name_yend=name_ye,\
                                  name_q=name_h, name_ds=name_ds))
                else:
                    #GM Spatial combo (KP 98-99)
                    output.append(GM_Combo(y=y, x=x, w=w,\
                                  name_y=name_y, name_x=name_x, name_ds=name_ds))
        if kp_het:
            for w in w_list:
                if endog:
                    #GM Spatial combo with het with non-spatial endogenous variables
                    kp_het_regs.append(GM_Combo_Het(y=y, x=x, w=w_list[0], yend=ye, q=h,\
                                         name_y=name_y, name_x=name_x, name_yend=name_ye,\
                                         name_q=name_h, name_ds=name_ds))
                else:
                    #GM Spatial combo with het
                    kp_het_regs.append(GM_Combo_Het(y=y, x=x, w=w_list[0],\
                                         name_y=name_y, name_x=name_x, name_ds=name_ds))
        output.extend(kp_het_regs)
        return output
    

def get_robust(reg, robust, wk=None):
    """Creates a new regression object, computes the robust standard errors,
    resets the regression object's internal cache and recompute the non-spatial 
    diagnostics.
    """
    USER.check_robust(robust, wk)    
    reg_robust = COPY.copy(reg)
    reg_robust._cache = {}
    reg_robust.vm = ROBUST.robust_vm(reg=reg_robust, wk=wk)
    reg_robust._get_diagnostics()
    return reg_robust


def _test():
    import doctest
    doctest.testmod()

if __name__ == '__main__':
    _test()




