import numpy as np
import copy as COPY
from ols import OLS
from twosls import TSLS
from twosls_sp import GM_Lag
from error_sp_het import GM_Error_Het, GM_Endog_Error_Het, GM_Combo_Het
from error_sp import GM_Endog_Error, GM_Error, GM_Combo
from error_sp_hom import GM_Endog_Error_Hom, GM_Error_Hom, GM_Combo_Hom
import robust as ROBUST
import user_output as USER

def spmodel(name_ds, w_list, wk_list, y, name_y, x, name_x, ye, name_ye,\
                h, name_h, r, name_r, s, name_s, t, name_t,\
                model_type,\
                spat_diag,\
                white, hac, kp_het,\
                sig2n_k_ols, sig2n_k_tsls, sig2n_k_gmlag,\
                max_iter, stop_crit, inf_lambda, comp_inverse, step1c,\
                instrument_lags, lag_user_inst,\
                vc_matrix, predy_resid,\
                ols_diag, moran,\
                ):
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
                  
    spat_diag   : boolean
                  Run spatial tests such as Moran's I on the residuals, LM and
                  AK tests

    white       : boolean
                  Compute White standard errors
    hac         : boolean
                  Compute HAC standard errors
    kp_het      : boolean
                  Run regression using KP2010 GMM approach for heteroskedasticity

    sig2n_k_ols : boolean
                  If True use n-k to compute OLS standard errors; if False use n
    sig2n_k_tsls : boolean
                   If True use N-k to compute 2SLS standard errors; if False use n
    sig2n_k_gmlag : boolean
                    If True use N-k to compute GM Lag standard errors; if False use n

    max_iter    : int
                  Maximum number of iterations to for improved GM efficiency
    stop_crit   : float
                  Stopping criterion for change in lambda (improved GM efficiency)
    inf_lambda  : boolean
                  If True compute inference on lambda (i.e. run the 'hom'
                  models); if False run the KP1998/1999 models
    comp_inverse : string
                   if 'Power exp' compute the inverse using the power
                   expansion; if 'True inv' compute the true inverse
    step1c      : boolean
                  If True compute Step 1c from Arraiz et al. (2010), if False
                  skip this step
    vc_matrix   : boolean
                  If True print VC matrix to screen in output, if False do
                  nothing
    predy_resid : string
                  Name of file to write output to, if None or False do nothing

    ols_diag    : boolean
                  Run nonspatial diagnostics (this might be removed later or
                  only applied to OLS)
    moran       : boolean
                  If True compute Moran's I of the residuals (only affects the
                  OLS case)

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
        model_type='Standard', ols_diag=True, spat_diag=True, moran=True,\
        vc_matrix=True, predy_resid=False,\
        max_iter=1, stop_crit=0.00001,\
        comp_inverse='Power_exp', step1c=False,\
        instrument_lags=1, lag_user_inst=True,\
        sig2n_k_ols=True, sig2n_k_tsls=False, sig2n_k_gmlag=False,\
        white=False, hac=False, kp_het=False, inf_lambda=False)
    >>> print reg[0].name_x
    ['CONSTANT', 'inc', 'hoval']
    >>> reg = spmodel(name_ds='columbus', w_list=[w], wk_list=[], y=y, name_y='crime', x=X, name_x=['inc', 'hoval'],\
        ye=[], name_ye=[], h=[], name_h=[],\
        r=None, name_r=None, s=None, name_s=None, t=None, name_t=None,\
        model_type='Standard', ols_diag=True, spat_diag=True, moran=False,\
        vc_matrix=False, predy_resid=False,\
        max_iter=1, stop_crit=0.00001,\
        comp_inverse='Power_exp', step1c=False,\
        instrument_lags=1, lag_user_inst=True,\
        sig2n_k_ols=True, sig2n_k_tsls=False, sig2n_k_gmlag=False,\
        white=True, hac=False, kp_het=False, inf_lambda=False)
    >>> print reg[0].name_x
    ['CONSTANT', 'inc', 'hoval']
    >>> reg = spmodel(name_ds='columbus', w_list=[w], wk_list=[wk], y=y, name_y='crime', x=X, name_x=['inc', 'hoval'],\
        ye=[], name_ye=[], h=[], name_h=[],\
        r=None, name_r=None, s=None, name_s=None, t=None, name_t=None,\
        model_type='Standard', ols_diag=True, spat_diag=False, moran=False,\
        vc_matrix=False, predy_resid=False,\
        max_iter=1, stop_crit=0.00001,\
        comp_inverse='Power_exp', step1c=False,\
        instrument_lags=1, lag_user_inst=True,\
        sig2n_k_ols=True, sig2n_k_tsls=False, sig2n_k_gmlag=False,\
        white=False, hac=True, kp_het=False, inf_lambda=False)
    >>> print reg[0].name_x
    ['CONSTANT', 'inc', 'hoval']
    >>> reg = spmodel(name_ds='columbus', w_list=[w], wk_list=[], y=y, name_y='crime', x=X, name_x=['inc', 'hoval'],\
        ye=[], name_ye=[], h=[], name_h=[],\
        r=None, name_r=None, s=None, name_s=None, t=None, name_t=None,\
        model_type='Spatial Lag', ols_diag=True, spat_diag=True, moran=False,\
        vc_matrix=True, predy_resid=False,\
        max_iter=1, stop_crit=0.00001,\
        comp_inverse='Power_exp', step1c=False,\
        instrument_lags=1, lag_user_inst=True,\
        sig2n_k_ols=True, sig2n_k_tsls=False, sig2n_k_gmlag=False,\
        white=False, hac=False, kp_het=False, inf_lambda=False)
    >>> print reg[0].name_z
    ['CONSTANT', 'inc', 'hoval', 'W_crime']
    >>> reg = spmodel(name_ds='columbus', w_list=[w], wk_list=[], y=y, name_y='crime', x=X, name_x=['inc', 'hoval'],\
        ye=[], name_ye=[], h=[], name_h=[],\
        r=None, name_r=None, s=None, name_s=None, t=None, name_t=None,\
        model_type='Spatial Lag', ols_diag=True, spat_diag=True, moran=False,\
        vc_matrix=True, predy_resid=False,\
        max_iter=1, stop_crit=0.00001,\
        comp_inverse='Power_exp', step1c=False,\
        instrument_lags=1, lag_user_inst=True,\
        sig2n_k_ols=True, sig2n_k_tsls=False, sig2n_k_gmlag=False,\
        white=True, hac=False, kp_het=False, inf_lambda=False)
    >>> print reg[0].name_z
    ['CONSTANT', 'inc', 'hoval', 'W_crime']
    >>> reg = spmodel(name_ds='columbus', w_list=[w], wk_list=[], y=y, name_y='crime', x=X, name_x=['inc', 'hoval'],\
        ye=[], name_ye=[], h=[], name_h=[],\
        r=None, name_r=None, s=None, name_s=None, t=None, name_t=None,\
        model_type='Spatial Lag', ols_diag=True, spat_diag=True, moran=False,\
        vc_matrix=False, predy_resid=False,\
        max_iter=1, stop_crit=0.00001,\
        comp_inverse='Power_exp', step1c=False,\
        instrument_lags=1, lag_user_inst=True,\
        sig2n_k_ols=True, sig2n_k_tsls=False, sig2n_k_gmlag=False,\
        white=False, hac=True, kp_het=False, inf_lambda=False)
    >>> print reg[0].name_z
    ['CONSTANT', 'inc', 'hoval', 'W_crime']
    >>> reg = spmodel(name_ds='columbus', w_list=[w], wk_list=[], y=y, name_y='crime', x=X, name_x=['inc', 'hoval'],\
        ye=[], name_ye=[], h=[], name_h=[],\
        r=None, name_r=None, s=None, name_s=None, t=None, name_t=None,\
        model_type='Spatial Error', ols_diag=True, spat_diag=False, moran=False,\
        vc_matrix=True, predy_resid=False,\
        max_iter=1, stop_crit=0.00001,\
        comp_inverse='Power_exp', step1c=False,\
        instrument_lags=1, lag_user_inst=True,\
        sig2n_k_ols=True, sig2n_k_tsls=False, sig2n_k_gmlag=False,\
        white=False, hac=False, kp_het=False, inf_lambda=False)
    >>> print reg[0].name_x
    ['CONSTANT', 'inc', 'hoval', 'lambda']
    >>> reg = spmodel(name_ds='columbus', w_list=[w], wk_list=[], y=y, name_y='crime', x=X, name_x=['inc', 'hoval'],\
        ye=[], name_ye=[], h=[], name_h=[],\
        r=None, name_r=None, s=None, name_s=None, t=None, name_t=None,\
        model_type='Spatial Error', ols_diag=True, spat_diag=True, moran=False,\
        vc_matrix=False, predy_resid=False,\
        max_iter=1, stop_crit=0.00001,\
        comp_inverse='Power_exp', step1c=False,\
        instrument_lags=1, lag_user_inst=True,\
        sig2n_k_ols=True, sig2n_k_tsls=False, sig2n_k_gmlag=False,\
        white=False, hac=False, kp_het=True, inf_lambda=False)
    >>> print reg[0].name_x
    ['CONSTANT', 'inc', 'hoval', 'lambda']
    >>> reg = spmodel(name_ds='columbus', w_list=[w], wk_list=[], y=y, name_y='crime', x=X, name_x=['inc', 'hoval'],\
        ye=[], name_ye=[], h=[], name_h=[],\
        r=None, name_r=None, s=None, name_s=None, t=None, name_t=None,\
        model_type='Spatial Error', ols_diag=True, spat_diag=False, moran=False,\
        vc_matrix=False, predy_resid=False,\
        max_iter=1, stop_crit=0.00001,\
        comp_inverse='Power_exp', step1c=False,\
        instrument_lags=1, lag_user_inst=True,\
        sig2n_k_ols=True, sig2n_k_tsls=False, sig2n_k_gmlag=False,\
        white=False, hac=False, kp_het=False, inf_lambda=True)
    >>> print reg[0].name_x
    ['CONSTANT', 'inc', 'hoval', 'lambda']
    >>> reg = spmodel(name_ds='columbus', w_list=[w], wk_list=[], y=y, name_y='crime', x=X, name_x=['inc', 'hoval'],\
        ye=[], name_ye=[], h=[], name_h=[],\
        r=None, name_r=None, s=None, name_s=None, t=None, name_t=None,\
        model_type='Spatial Error', ols_diag=True, spat_diag=False, moran=False,\
        vc_matrix=False, predy_resid=False,\
        max_iter=1, stop_crit=0.00001,\
        comp_inverse='Power_exp', step1c=False,\
        instrument_lags=1, lag_user_inst=True,\
        sig2n_k_ols=True, sig2n_k_tsls=False, sig2n_k_gmlag=False,\
        white=False, hac=False, kp_het=True, inf_lambda=True)
    >>> print reg[0].name_x
    ['CONSTANT', 'inc', 'hoval', 'lambda']
    >>> reg = spmodel(name_ds='columbus', w_list=[w], wk_list=[], y=y, name_y='crime', x=X, name_x=['inc', 'hoval'],\
        ye=[], name_ye=[], h=[], name_h=[],\
        r=None, name_r=None, s=None, name_s=None, t=None, name_t=None,\
        model_type='Spatial Lag+Error', ols_diag=True, spat_diag=False, moran=False,\
        vc_matrix=False, predy_resid=False,\
        max_iter=1, stop_crit=0.00001,\
        comp_inverse='Power_exp', step1c=False,\
        instrument_lags=1, lag_user_inst=True,\
        sig2n_k_ols=True, sig2n_k_tsls=False, sig2n_k_gmlag=False,\
        white=False, hac=False, kp_het=False, inf_lambda=False)
    >>> print reg[0].name_z
    ['CONSTANT', 'inc', 'hoval', 'W_crime', 'lambda']
    >>> reg = spmodel(name_ds='columbus', w_list=[w], wk_list=[], y=y, name_y='crime', x=X, name_x=['inc', 'hoval'],\
        ye=[], name_ye=[], h=[], name_h=[],\
        r=None, name_r=None, s=None, name_s=None, t=None, name_t=None,\
        model_type='Spatial Lag+Error', ols_diag=True, spat_diag=False, moran=False,\
        vc_matrix=True, predy_resid=False,\
        max_iter=1, stop_crit=0.00001,\
        comp_inverse='Power_exp', step1c=False,\
        instrument_lags=1, lag_user_inst=True,\
        sig2n_k_ols=True, sig2n_k_tsls=False, sig2n_k_gmlag=False,\
        white=False, hac=False, kp_het=True, inf_lambda=False)
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
        model_type='Standard', ols_diag=True, spat_diag=True, moran=False,\
        vc_matrix=True, predy_resid=False,\
        max_iter=1, stop_crit=0.00001,\
        comp_inverse='Power_exp', step1c=False,\
        instrument_lags=1, lag_user_inst=True,\
        sig2n_k_ols=True, sig2n_k_tsls=False, sig2n_k_gmlag=False,\
        white=False, hac=False, kp_het=False, inf_lambda=False)
    >>> print reg[0].name_z
    ['CONSTANT', 'inc', 'hoval']
    >>> reg = spmodel(name_ds='columbus', w_list=[w], wk_list=[], y=y, name_y='crime', x=X, name_x=['inc'],\
        ye=yd, name_ye=['hoval'], h=q, name_h=['discbd'],\
        r=None, name_r=None, s=None, name_s=None, t=None, name_t=None,\
        model_type='Standard', ols_diag=True, spat_diag=False, moran=False,\
        vc_matrix=False, predy_resid=False,\
        max_iter=1, stop_crit=0.00001,\
        comp_inverse='Power_exp', step1c=False,\
        instrument_lags=1, lag_user_inst=True,\
        sig2n_k_ols=True, sig2n_k_tsls=False, sig2n_k_gmlag=False,\
        white=True, hac=False, kp_het=False, inf_lambda=False)
    >>> print reg[0].name_z
    ['CONSTANT', 'inc', 'hoval']
    >>> reg = spmodel(name_ds='columbus', w_list=[w], wk_list=[wk], y=y, name_y='crime', x=X, name_x=['inc'],\
        ye=yd, name_ye=['hoval'], h=q, name_h=['discbd'],\
        r=None, name_r=None, s=None, name_s=None, t=None, name_t=None,\
        model_type='Standard', ols_diag=True, spat_diag=True, moran=False,\
        vc_matrix=True, predy_resid=False,\
        max_iter=1, stop_crit=0.00001,\
        comp_inverse='Power_exp', step1c=False,\
        instrument_lags=1, lag_user_inst=True,\
        sig2n_k_ols=True, sig2n_k_tsls=False, sig2n_k_gmlag=False,\
        white=False, hac=True, kp_het=False, inf_lambda=False)
    >>> print reg[0].name_z
    ['CONSTANT', 'inc', 'hoval']
    >>> reg = spmodel(name_ds='columbus', w_list=[w], wk_list=[], y=y, name_y='crime', x=X, name_x=['inc'],\
        ye=yd, name_ye=['hoval'], h=q, name_h=['discbd'],\
        r=None, name_r=None, s=None, name_s=None, t=None, name_t=None,\
        model_type='Spatial Lag', ols_diag=True, spat_diag=False, moran=False,\
        vc_matrix=False, predy_resid=False,\
        max_iter=1, stop_crit=0.00001,\
        comp_inverse='Power_exp', step1c=False,\
        instrument_lags=1, lag_user_inst=True,\
        sig2n_k_ols=True, sig2n_k_tsls=False, sig2n_k_gmlag=False,\
        white=False, hac=False, kp_het=False, inf_lambda=False)
    >>> print reg[0].name_z
    ['CONSTANT', 'inc', 'hoval', 'W_crime']
    >>> reg = spmodel(name_ds='columbus', w_list=[w], wk_list=[], y=y, name_y='crime', x=X, name_x=['inc'],\
        ye=yd, name_ye=['hoval'], h=q, name_h=['discbd'],\
        r=None, name_r=None, s=None, name_s=None, t=None, name_t=None,\
        model_type='Spatial Lag', ols_diag=True, spat_diag=False, moran=False,\
        vc_matrix=False, predy_resid=False,\
        max_iter=1, stop_crit=0.00001,\
        comp_inverse='Power_exp', step1c=False,\
        instrument_lags=1, lag_user_inst=True,\
        sig2n_k_ols=True, sig2n_k_tsls=False, sig2n_k_gmlag=False,\
        white=True, hac=False, kp_het=False, inf_lambda=False)
    >>> print reg[0].name_z
    ['CONSTANT', 'inc', 'hoval', 'W_crime']
    >>> reg = spmodel(name_ds='columbus', w_list=[w], wk_list=[], y=y, name_y='crime', x=X, name_x=['inc'],\
        ye=yd, name_ye=['hoval'], h=q, name_h=['discbd'],\
        r=None, name_r=None, s=None, name_s=None, t=None, name_t=None,\
        model_type='Spatial Lag', ols_diag=True, spat_diag=False, moran=False,\
        vc_matrix=False, predy_resid=False,\
        max_iter=1, stop_crit=0.00001,\
        comp_inverse='Power_exp', step1c=False,\
        instrument_lags=1, lag_user_inst=True,\
        sig2n_k_ols=True, sig2n_k_tsls=False, sig2n_k_gmlag=False,\
        white=False, hac=True, kp_het=False, inf_lambda=False)
    >>> print reg[0].name_z
    ['CONSTANT', 'inc', 'hoval', 'W_crime']
    >>> reg = spmodel(name_ds='columbus', w_list=[w], wk_list=[], y=y, name_y='crime', x=X, name_x=['inc'],\
        ye=yd, name_ye=['hoval'], h=q, name_h=['discbd'],\
        r=None, name_r=None, s=None, name_s=None, t=None, name_t=None,\
        model_type='Spatial Error', ols_diag=True, spat_diag=True, moran=False,\
        vc_matrix=True, predy_resid=False,\
        max_iter=1, stop_crit=0.00001,\
        comp_inverse='Power_exp', step1c=False,\
        instrument_lags=1, lag_user_inst=True,\
        sig2n_k_ols=True, sig2n_k_tsls=False, sig2n_k_gmlag=False,\
        white=False, hac=False, kp_het=False, inf_lambda=False)
    >>> print reg[0].name_z
    ['CONSTANT', 'inc', 'hoval', 'lambda']
    >>> reg = spmodel(name_ds='columbus', w_list=[w], wk_list=[], y=y, name_y='crime', x=X, name_x=['inc'],\
        ye=yd, name_ye=['hoval'], h=q, name_h=['discbd'],\
        r=None, name_r=None, s=None, name_s=None, t=None, name_t=None,\
        model_type='Spatial Error', ols_diag=True, spat_diag=True, moran=False,\
        vc_matrix=False, predy_resid=False,\
        max_iter=1, stop_crit=0.00001,\
        comp_inverse='Power_exp', step1c=False,\
        instrument_lags=1, lag_user_inst=True,\
        sig2n_k_ols=True, sig2n_k_tsls=False, sig2n_k_gmlag=False,\
        white=False, hac=False, kp_het=True, inf_lambda=False)
    >>> print reg[0].name_z
    ['CONSTANT', 'inc', 'hoval', 'lambda']
    >>> reg = spmodel(name_ds='columbus', w_list=[w], wk_list=[], y=y, name_y='crime', x=X, name_x=['inc'],\
        ye=yd, name_ye=['hoval'], h=q, name_h=['discbd'],\
        r=None, name_r=None, s=None, name_s=None, t=None, name_t=None,\
        model_type='Spatial Lag+Error', ols_diag=True, spat_diag=False, moran=False,\
        vc_matrix=False, predy_resid=False,\
        max_iter=1, stop_crit=0.00001,\
        comp_inverse='Power_exp', step1c=False,\
        instrument_lags=1, lag_user_inst=True,\
        sig2n_k_ols=True, sig2n_k_tsls=False, sig2n_k_gmlag=False,\
        white=False, hac=False, kp_het=False, inf_lambda=False)
    >>> print reg[0].name_z
    ['CONSTANT', 'inc', 'hoval', 'W_crime', 'lambda']
    >>> reg = spmodel(name_ds='columbus', w_list=[w], wk_list=[], y=y, name_y='crime', x=X, name_x=['inc'],\
        ye=yd, name_ye=['hoval'], h=q, name_h=['discbd'],\
        r=None, name_r=None, s=None, name_s=None, t=None, name_t=None,\
        model_type='Spatial Lag+Error', ols_diag=True, spat_diag=False, moran=False,\
        vc_matrix=True, predy_resid=False,\
        max_iter=1, stop_crit=0.00001,\
        comp_inverse='Power_exp', step1c=False,\
        instrument_lags=1, lag_user_inst=True,\
        sig2n_k_ols=True, sig2n_k_tsls=False, sig2n_k_gmlag=False,\
        white=False, hac=False, kp_het=True, inf_lambda=False)
    >>> print reg[0].name_z
    ['CONSTANT', 'inc', 'hoval', 'W_crime', 'lambda']

    """
    output = []
    if predy_resid:
        pred_res = y
        pred_res.shape = (y.shape[0], 1)
        header_pr = name_y



    if model_type == 'Standard':
        if name_ye:
            # run 2SLS regression; with or without nonspatial diagnostics
            reg = TSLS(y=y, x=x, yend=ye, q=h,\
                        nonspat_diag=ols_diag, spat_diag=False, vm=False,\
                        name_y=name_y, name_x=name_x, name_yend=name_ye,\
                        name_q=name_h, name_ds=name_ds, sig2n_k=sig2n_k_tsls)
        else:
            # run OLS regression; with or without nonspatial diagnostics
            reg = OLS(y=y, x=x,\
                       nonspat_diag=ols_diag, spat_diag=False, vm=False,\
                       name_y=name_y, name_x=name_x, name_ds=name_ds,
                       sig2n_k=sig2n_k_ols)
        if predy_resid:  # write out predicted values and residuals
            pred_res = np.hstack((pred_res, reg.predy, reg.u))
            header_pr += ',standard_predy,standard_resid'
        if w_list and spat_diag:
            for w in w_list:
                # add spatial diagnostics for each W
                reg_spat = COPY.copy(reg)
                reg_spat._get_diagnostics(w=w, beta_diag=False, moran=moran,\
                                          nonspat_diag=False, spat_diag=True,
                                          vm=vc_matrix)
                output.append(reg_spat)
        else:
            if vc_matrix:
                reg._get_diagnostics(beta_diag=False,\
                                          nonspat_diag=False, spat_diag=False,
                                          vm=vc_matrix)
            output.append(reg)
        if white:
            # compute White std errors
            output.append(get_robust(reg, 'white'))
        if hac:
            for wk in wk_list:
                # compute HAC std errors
                output.append(get_robust(reg, 'hac', wk))



    elif model_type == 'Spatial Lag':
        counter = 1
        for w in w_list:
            if name_ye:
               #GM Spatial lag with non-spatial endog variables
                reg = GM_Lag(y=y, x=x, w=w, yend=ye, q=h, vm=vc_matrix,\
                      w_lags=instrument_lags, lag_q=lag_user_inst,\
                      name_y=name_y, name_x=name_x, name_yend=name_ye,\
                      name_q=name_h, name_ds=name_ds, sig2n_k=sig2n_k_gmlag,\
                      spat_diag=spat_diag)
            else:
               #GM Spatial lag
                reg = GM_Lag(y=y, x=x, w=w, vm=vc_matrix,\
                      w_lags=instrument_lags, lag_q=lag_user_inst,\
                      name_y=name_y, name_x=name_x, name_ds=name_ds,\
                      sig2n_k=sig2n_k_gmlag, spat_diag=spat_diag)
            if predy_resid:  # write out predicted values and residuals
                pred_res, header_pr, counter = collect_predy_resid(\
                                       pred_res, header_pr, reg, '',\
                                       True, len(w_list), counter)
            output.append(reg)

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



    elif model_type == 'Spatial Error':
        counter = 1
        if inf_lambda:
            for w in w_list:
                if name_ye:
                    #GM Spatial error (hom) with non-spatial endogenous variable
                    reg = GM_Endog_Error_Hom(y=y, x=x, w=w, yend=ye, q=h, vm=vc_matrix,\
                          max_iter=max_iter, epsilon=stop_crit,\
                          name_y=name_y, name_x=name_x, name_yend=name_ye,\
                          name_q=name_h, name_ds=name_ds)
                else:
                    #GM Spatial error (hom) 
                    reg = GM_Error_Hom(y=y, x=x, w=w, vm=vc_matrix,\
                          max_iter=max_iter, epsilon=stop_crit,\
                          name_y=name_y, name_x=name_x, name_ds=name_ds)
                if predy_resid:  # write out predicted values and residuals
                    pred_res, header_pr, counter = collect_predy_resid(\
                                           pred_res, header_pr, reg, '',\
                                           False, len(w_list), counter)
                output.append(reg)
        else:
            for w in w_list:
                if name_ye:
                    #GM Spatial error with non-spatial endogenous variable (KP 98-99)
                    reg = GM_Endog_Error(y=y, x=x, w=w, yend=ye, q=h, vm=vc_matrix,\
                          name_y=name_y, name_x=name_x, name_yend=name_ye,\
                          name_q=name_h, name_ds=name_ds)
                else:
                    #GM Spatial error (KP 98-99)
                    reg = GM_Error(y=y, x=x, w=w, vm=vc_matrix,\
                          name_y=name_y, name_x=name_x, name_ds=name_ds)
                if predy_resid:  # write out predicted values and residuals
                    pred_res, header_pr, counter = collect_predy_resid(\
                                           pred_res, header_pr, reg, '',\
                                           False, len(w_list), counter)
                output.append(reg)

        counter = 1
        if kp_het:
            for w in w_list:
                if name_ye:
                    #GM Spatial error with het and with non-spatial endogenous variable
                    reg = GM_Endog_Error_Het(y=y, x=x, w=w, yend=ye, q=h, vm=vc_matrix,\
                          max_iter=max_iter, epsilon=stop_crit,\
                          step1c=step1c, inv_method=comp_inverse,\
                          name_y=name_y, name_x=name_x, name_yend=name_ye,\
                          name_q=name_h, name_ds=name_ds)
                else:
                    #GM Spatial error with het
                    reg = GM_Error_Het(y=y, x=x, w=w, vm=vc_matrix,\
                          max_iter=max_iter, epsilon=stop_crit,\
                          step1c=step1c,\
                          name_y=name_y, name_x=name_x, name_ds=name_ds)
                if predy_resid:  # write out predicted values and residuals
                    pred_res, header_pr, counter = collect_predy_resid(\
                                           pred_res, header_pr, reg, 'het_',\
                                           False, len(w_list), counter)
                output.append(reg)



    elif model_type == 'Spatial Lag+Error':
        counter = 1
        if inf_lambda:
            for w in w_list:
                if name_ye:
                    #GM Spatial combo (hom) with non-spatial endogenous variables
                    reg = GM_Combo_Hom(y=y, x=x, w=w, yend=ye, q=h, vm=vc_matrix,\
                          w_lags=instrument_lags, lag_q=lag_user_inst,\
                          max_iter=max_iter, epsilon=stop_crit,\
                          name_y=name_y, name_x=name_x, name_yend=name_ye,\
                          name_q=name_h, name_ds=name_ds)
                else:
                    #GM Spatial combo (hom)
                    reg = GM_Combo_Hom(y=y, x=x, w=w, vm=vc_matrix,\
                          w_lags=instrument_lags, lag_q=lag_user_inst,\
                          max_iter=max_iter, epsilon=stop_crit,\
                          name_y=name_y, name_x=name_x, name_ds=name_ds)
                if predy_resid:  # write out predicted values and residuals
                    pred_res, header_pr, counter = collect_predy_resid(\
                                           pred_res, header_pr, reg, '',\
                                           True, len(w_list), counter)
                output.append(reg)
        else:
            for w in w_list:
                if name_ye:
                    #GM Spatial combo with non-spatial endogenous variables (KP 98-99)
                    reg = GM_Combo(y=y, x=x, w=w, yend=ye, q=h, vm=vc_matrix,\
                          w_lags=instrument_lags, lag_q=lag_user_inst,\
                          name_y=name_y, name_x=name_x, name_yend=name_ye,\
                          name_q=name_h, name_ds=name_ds)
                else:
                    #GM Spatial combo (KP 98-99)
                    reg = GM_Combo(y=y, x=x, w=w, vm=vc_matrix,\
                          w_lags=instrument_lags, lag_q=lag_user_inst,\
                          name_y=name_y, name_x=name_x, name_ds=name_ds)
                if predy_resid:  # write out predicted values and residuals
                    pred_res, header_pr, counter = collect_predy_resid(\
                                           pred_res, header_pr, reg, '',\
                                           True, len(w_list), counter)
                output.append(reg)

        counter = 1
        if kp_het:
            for w in w_list:
                if name_ye:
                    #GM Spatial combo with het with non-spatial endogenous variables
                    reg = GM_Combo_Het(y=y, x=x, w=w_list[0], yend=ye, q=h, vm=vc_matrix,\
                          w_lags=instrument_lags, lag_q=lag_user_inst,\
                          max_iter=max_iter, epsilon=stop_crit,\
                          step1c=step1c, inv_method=comp_inverse,\
                          name_y=name_y, name_x=name_x, name_yend=name_ye,\
                          name_q=name_h, name_ds=name_ds)
                else:
                    #GM Spatial combo with het
                    reg = GM_Combo_Het(y=y, x=x, w=w_list[0], vm=vc_matrix,\
                          w_lags=instrument_lags, lag_q=lag_user_inst,\
                          max_iter=max_iter, epsilon=stop_crit,\
                          step1c=step1c, inv_method=comp_inverse,\
                          name_y=name_y, name_x=name_x, name_ds=name_ds)
                if predy_resid:  # write out predicted values and residuals
                    pred_res, header_pr, counter = collect_predy_resid(\
                                           pred_res, header_pr, reg, 'het',\
                                           True, len(w_list), counter)
                output.append(reg)


    if predy_resid:
        outfile = open(predy_resid, 'w')
        outfile.write(header_pr+'\n')
        np.savetxt(outfile, pred_res, delimiter=',')
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


def collect_predy_resid(pred_res, header_pr, reg, model, spatial, ws, counter):
    if ws > 1:
        lead = model+'W'+str(counter)+'_'
        counter += 1
    else:
        lead = model
    if spatial:
        pred_res = np.hstack((pred_res, reg.predy, reg.u, reg.predy_sp, reg.resid_sp))
        header_pr += ','+lead+'predy,'+lead+'resid,'+lead+'predy_sp,'+lead+'resid_sp'
    else:
        pred_res = np.hstack((pred_res, reg.predy, reg.u))
        header_pr += ','+lead+'predy,'+lead+'resid'
    return pred_res, header_pr, counter
    



def _test():
    import doctest
    doctest.testmod()

if __name__ == '__main__':
    _test()




