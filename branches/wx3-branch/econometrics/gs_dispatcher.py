import numpy as np
import copy as COPY
from ols import OLS
from twosls import TSLS
from twosls_sp import GM_Lag
from error_sp_het import GM_Error_Het, GM_Endog_Error_Het, GM_Combo_Het
from error_sp import GM_Endog_Error, GM_Error, GM_Combo
from error_sp_hom import GM_Endog_Error_Hom, GM_Error_Hom, GM_Combo_Hom
from ols_regimes import OLS_Regimes
from twosls_regimes import TSLS_Regimes
from twosls_sp_regimes import GM_Lag_Regimes
from error_sp_het_regimes import GM_Error_Het_Regimes, GM_Endog_Error_Het_Regimes, GM_Combo_Het_Regimes
from error_sp_regimes import GM_Endog_Error_Regimes, GM_Error_Regimes, GM_Combo_Regimes
from error_sp_hom_regimes import GM_Endog_Error_Hom_Regimes, GM_Error_Hom_Regimes, GM_Combo_Hom_Regimes
from ml_error import ML_Error
from ml_error_regimes import ML_Error_Regimes
from ml_lag import ML_Lag
from ml_lag_regimes import ML_Lag_Regimes
import robust as ROBUST
import summary_output as SUMMARY
import user_output as USER
import regimes as REGI

INV_METHODS = ("Power Expansion", "True Inverse",)
ML_METHODS = ("Full", "Ord",)


class Spmodel:
    """
    A single class to call all the econometric models in pysal.

    Parameters
    ----------

    name_ds     : string
                  Name of dataset for use in output, for description only.
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
    r           : list
                  List of length n with regimes variable
    name_r      : string
                  Name of regime variable for use in output
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
    white_test  : boolean
                  If True, compute White's specification robust test.
                  (only affects the OLS case and requires nonspat_diag=True)
    regime_err_sep  : boolean
                      If True, a separate regression is run for each regime.
    regime_lag_sep  : boolean
                      If True, the spatial parameter for spatial lag is also
                      computed according to different regimes. If False (default),
                      the spatial parameter is fixed accross regimes.
    cores       : int
                  Amount of cores to be used for multiprocessing tasks.
                  Default: None, which is same as maximum number possible.
    method      : string
                  If 'gm' computes models using GM estimators. If 'ml' uses
                  maximum-likelihood. If 'ols', computes OLS.
    ml_diag     : boolean
                  Run diagnostics for ML models
    ml_method   : string
                  if 'full', brute force calculation in ML (full matrix expressions)
                  if 'ord', use Ord's method in ML
    ml_epsilon  : float
                  Tolerance criterion for ML models          

    Returns
    -------
    return  : PySAL regression object

    Examples
    --------

    Not all combinations are tested below

    >>> import numpy as np
    >>> import pysal
    >>> w = pysal.rook_from_shapefile("examples/columbus.shp")
    >>> w.name = 'Rook columbus weights'
    >>> w.transform = 'r'
    >>> w2 = pysal.queen_from_shapefile("examples/columbus.shp")
    >>> w2.name = 'Queen columbus weights'
    >>> w2.transform = 'r'
    >>> wk = pysal.kernelW_from_shapefile("examples/columbus.shp",k=4,function='triangular',idVariable=None,fixed=False)
    >>> wk.name = 'Kernel columbus weights'
    >>> db=pysal.open("examples/columbus.dbf","r")
    >>> y = np.array(db.by_col("CRIME"))
    >>> y = np.reshape(y, (49,1))
    >>> # no non-spatial endogenous variables
    >>> X = []
    >>> X.append(db.by_col("INC"))
    >>> X.append(db.by_col("HOVAL"))
    >>> X = np.array(X).T

    No non-spatial endogenous variables

    >>> reg = Spmodel(name_ds='columbus', w_list=[w, w2], wk_list=[], y=y, name_y='crime', x=X, name_x=['inc', 'hoval'],\
        ye=[], name_ye=[], h=[], name_h=[],\
        r=None, name_r=None, s=None, name_s=None, t=None, name_t=None,\
        model_type='Standard', ols_diag=True, spat_diag=True, moran=True,\
        white_test=True, vc_matrix=True, predy_resid=False,\
        max_iter=1, stop_crit=0.00001,\
        comp_inverse='Power Expansion', step1c=False,\
        instrument_lags=1, lag_user_inst=True,\
        sig2n_k_ols=True, sig2n_k_tsls=False, sig2n_k_gmlag=False,\
        regime_err_sep=None, regime_lag_sep=None, cores=None,\
        white=False, hac=False, kp_het=False, inf_lambda=False, method='ols',\
        ml_diag=False,ml_method='full',ml_epsilon=0.00001)
    >>> print reg.output[1].name_x
    ['CONSTANT', 'inc', 'hoval']
    >>> reg = Spmodel(name_ds='columbus', w_list=[w, w2], wk_list=[], y=y, name_y='crime', x=X, name_x=['inc', 'hoval'],\
        ye=[], name_ye=[], h=[], name_h=[],\
        r=None, name_r=None, s=None, name_s=None, t=None, name_t=None,\
        model_type='Standard', ols_diag=True, spat_diag=True, moran=False,\
        white_test=False, vc_matrix=False, predy_resid=False,\
        max_iter=1, stop_crit=0.00001,\
        comp_inverse='Power Expansion', step1c=False,\
        instrument_lags=1, lag_user_inst=True,\
        sig2n_k_ols=True, sig2n_k_tsls=False, sig2n_k_gmlag=False,\
        regime_err_sep=None, regime_lag_sep=None, cores=None,\
        white=True, hac=False, kp_het=False, inf_lambda=False, method='ols',\
        ml_diag=False,ml_method='full',ml_epsilon=0.00001)
    >>> print reg.output[1].name_x
    ['CONSTANT', 'inc', 'hoval']
    >>> reg = Spmodel(name_ds='columbus', w_list=[w, w2], wk_list=[wk], y=y, name_y='crime', x=X, name_x=['inc', 'hoval'],\
        ye=[], name_ye=[], h=[], name_h=[],\
        r=None, name_r=None, s=None, name_s=None, t=None, name_t=None,\
        model_type='Standard', ols_diag=True, spat_diag=False, moran=False,\
        white_test=False, vc_matrix=False, predy_resid=False,\
        max_iter=1, stop_crit=0.00001,\
        comp_inverse='Power Expansion', step1c=False,\
        instrument_lags=1, lag_user_inst=True,\
        sig2n_k_ols=True, sig2n_k_tsls=False, sig2n_k_gmlag=False,\
        regime_err_sep=None, regime_lag_sep=None, cores=None,\
        white=False, hac=True, kp_het=False, inf_lambda=False, method='ols',\
        ml_diag=False,ml_method='full',ml_epsilon=0.00001)
    >>> print reg.output[1].name_x
    ['CONSTANT', 'inc', 'hoval']
    >>> reg = Spmodel(name_ds='columbus', w_list=[w, w2], wk_list=[], y=y, name_y='crime', x=X, name_x=['inc', 'hoval'],\
        ye=[], name_ye=[], h=[], name_h=[],\
        r=None, name_r=None, s=None, name_s=None, t=None, name_t=None,\
        model_type='Spatial Lag', ols_diag=True, spat_diag=True, moran=False,\
        white_test=False, vc_matrix=True, predy_resid=False,\
        max_iter=1, stop_crit=0.00001,\
        comp_inverse='Power Expansion', step1c=False,\
        instrument_lags=1, lag_user_inst=True,\
        sig2n_k_ols=True, sig2n_k_tsls=False, sig2n_k_gmlag=False,\
        regime_err_sep=None, regime_lag_sep=None, cores=None,\
        white=False, hac=False, kp_het=False, inf_lambda=False, method='gm',\
        ml_diag=False,ml_method='full',ml_epsilon=0.00001)
    >>> print reg.output[1].name_z
    ['CONSTANT', 'inc', 'hoval', 'W_crime']
    >>> reg = Spmodel(name_ds='columbus', w_list=[w, w2], wk_list=[], y=y, name_y='crime', x=X, name_x=['inc', 'hoval'],\
        ye=[], name_ye=[], h=[], name_h=[],\
        r=None, name_r=None, s=None, name_s=None, t=None, name_t=None,\
        model_type='Spatial Lag', ols_diag=True, spat_diag=True, moran=False,\
        white_test=False, vc_matrix=True, predy_resid=False,\
        max_iter=1, stop_crit=0.00001,\
        comp_inverse='Power Expansion', step1c=False,\
        instrument_lags=1, lag_user_inst=True,\
        sig2n_k_ols=True, sig2n_k_tsls=False, sig2n_k_gmlag=False,\
        regime_err_sep=None, regime_lag_sep=None, cores=None,\
        white=True, hac=False, kp_het=False, inf_lambda=False, method='gm',\
        ml_diag=False,ml_method='full',ml_epsilon=0.00001)
    >>> print reg.output[1].name_z
    ['CONSTANT', 'inc', 'hoval', 'W_crime']
    >>> reg = Spmodel(name_ds='columbus', w_list=[w, w2], wk_list=[], y=y, name_y='crime', x=X, name_x=['inc', 'hoval'],\
        ye=[], name_ye=[], h=[], name_h=[],\
        r=None, name_r=None, s=None, name_s=None, t=None, name_t=None,\
        model_type='Spatial Lag', ols_diag=True, spat_diag=True, moran=False,\
        white_test=False, vc_matrix=False, predy_resid=False,\
        max_iter=1, stop_crit=0.00001,\
        comp_inverse='Power Expansion', step1c=False,\
        instrument_lags=1, lag_user_inst=True,\
        sig2n_k_ols=True, sig2n_k_tsls=False, sig2n_k_gmlag=False,\
        regime_err_sep=None, regime_lag_sep=None, cores=None,\
        white=False, hac=False, kp_het=False, inf_lambda=False, method='gm',\
        ml_diag=False,ml_method='full',ml_epsilon=0.00001)
    >>> print reg.output[1].name_z
    ['CONSTANT', 'inc', 'hoval', 'W_crime']
    >>> reg = Spmodel(name_ds='columbus', w_list=[w, w2], wk_list=[], y=y, name_y='crime', x=X, name_x=['inc', 'hoval'],\
        ye=[], name_ye=[], h=[], name_h=[],\
        r=None, name_r=None, s=None, name_s=None, t=None, name_t=None,\
        model_type='Spatial Error', ols_diag=True, spat_diag=False, moran=False,\
        white_test=False, vc_matrix=True, predy_resid=False,\
        max_iter=1, stop_crit=0.00001,\
        comp_inverse='Power Expansion', step1c=False,\
        instrument_lags=1, lag_user_inst=True,\
        sig2n_k_ols=True, sig2n_k_tsls=False, sig2n_k_gmlag=False,\
        regime_err_sep=None, regime_lag_sep=None, cores=None,\
        white=False, hac=False, kp_het=False, inf_lambda=False, method='gm',\
        ml_diag=False,ml_method='full',ml_epsilon=0.00001)
    >>> print reg.output[1].name_x
    ['CONSTANT', 'inc', 'hoval', 'lambda']
    >>> reg = Spmodel(name_ds='columbus', w_list=[w, w2], wk_list=[], y=y, name_y='crime', x=X, name_x=['inc', 'hoval'],\
        ye=[], name_ye=[], h=[], name_h=[],\
        r=None, name_r=None, s=None, name_s=None, t=None, name_t=None,\
        model_type='Spatial Error', ols_diag=True, spat_diag=True, moran=False,\
        white_test=False, vc_matrix=False, predy_resid=False,\
        max_iter=1, stop_crit=0.00001,\
        comp_inverse='Power Expansion', step1c=False,\
        instrument_lags=1, lag_user_inst=True,\
        sig2n_k_ols=True, sig2n_k_tsls=False, sig2n_k_gmlag=False,\
        regime_err_sep=None, regime_lag_sep=None, cores=None,\
        white=False, hac=False, kp_het=True, inf_lambda=False, method='gm',\
        ml_diag=False,ml_method='full',ml_epsilon=0.00001)
    >>> print reg.output[1].name_x
    ['CONSTANT', 'inc', 'hoval', 'lambda']
    >>> reg = Spmodel(name_ds='columbus', w_list=[w, w2], wk_list=[], y=y, name_y='crime', x=X, name_x=['inc', 'hoval'],\
        ye=[], name_ye=[], h=[], name_h=[],\
        r=None, name_r=None, s=None, name_s=None, t=None, name_t=None,\
        model_type='Spatial Error', ols_diag=True, spat_diag=False, moran=False,\
        white_test=False, vc_matrix=False, predy_resid=False,\
        max_iter=1, stop_crit=0.00001,\
        comp_inverse='Power Expansion', step1c=False,\
        instrument_lags=1, lag_user_inst=True,\
        sig2n_k_ols=True, sig2n_k_tsls=False, sig2n_k_gmlag=False,\
        regime_err_sep=None, regime_lag_sep=None, cores=None,\
        white=False, hac=False, kp_het=False, inf_lambda=True, method='gm',\
        ml_diag=False,ml_method='full',ml_epsilon=0.00001)
    >>> print reg.output[1].name_x
    ['CONSTANT', 'inc', 'hoval', 'lambda']
    >>> reg = Spmodel(name_ds='columbus', w_list=[w, w2], wk_list=[], y=y, name_y='crime', x=X, name_x=['inc', 'hoval'],\
        ye=[], name_ye=[], h=[], name_h=[],\
        r=None, name_r=None, s=None, name_s=None, t=None, name_t=None,\
        model_type='Spatial Error', ols_diag=True, spat_diag=False, moran=False,\
        white_test=False, vc_matrix=False, predy_resid=False,\
        max_iter=1, stop_crit=0.00001,\
        comp_inverse='Power Expansion', step1c=False,\
        instrument_lags=1, lag_user_inst=True,\
        sig2n_k_ols=True, sig2n_k_tsls=False, sig2n_k_gmlag=False,\
        regime_err_sep=None, regime_lag_sep=None, cores=None,\
        white=False, hac=False, kp_het=True, inf_lambda=True, method='gm',\
        ml_diag=False,ml_method='full',ml_epsilon=0.00001)
    >>> print reg.output[1].name_x
    ['CONSTANT', 'inc', 'hoval', 'lambda']
    >>> reg = Spmodel(name_ds='columbus', w_list=[w, w2], wk_list=[], y=y, name_y='crime', x=X, name_x=['inc', 'hoval'],\
        ye=[], name_ye=[], h=[], name_h=[],\
        r=None, name_r=None, s=None, name_s=None, t=None, name_t=None,\
        model_type='Spatial Lag+Error', ols_diag=True, spat_diag=False, moran=False,\
        white_test=False, vc_matrix=False, predy_resid=False,\
        max_iter=1, stop_crit=0.00001,\
        comp_inverse='Power Expansion', step1c=False,\
        instrument_lags=1, lag_user_inst=True,\
        sig2n_k_ols=True, sig2n_k_tsls=False, sig2n_k_gmlag=False,\
        regime_err_sep=None, regime_lag_sep=None, cores=None,\
        white=False, hac=False, kp_het=False, inf_lambda=False, method='gm',\
        ml_diag=False,ml_method='full',ml_epsilon=0.00001)
    >>> print reg.output[1].name_z
    ['CONSTANT', 'inc', 'hoval', 'W_crime', 'lambda']
    >>> reg = Spmodel(name_ds='columbus', w_list=[w, w2], wk_list=[], y=y, name_y='crime', x=X, name_x=['inc', 'hoval'],\
        ye=[], name_ye=[], h=[], name_h=[],\
        r=None, name_r=None, s=None, name_s=None, t=None, name_t=None,\
        model_type='Spatial Lag+Error', ols_diag=True, spat_diag=False, moran=False,\
        white_test=False, vc_matrix=True, predy_resid=False,\
        max_iter=1, stop_crit=0.00001,\
        comp_inverse='Power Expansion', step1c=False,\
        instrument_lags=1, lag_user_inst=True,\
        sig2n_k_ols=True, sig2n_k_tsls=False, sig2n_k_gmlag=False,\
        regime_err_sep=None, regime_lag_sep=None, cores=None,\
        white=False, hac=False, kp_het=True, inf_lambda=False, method='gm',\
        ml_diag=False,ml_method='full',ml_epsilon=0.00001)
    >>> print reg.output[1].name_z
    ['CONSTANT', 'inc', 'hoval', 'W_crime', 'lambda']

    Add in non-spatial endogenous variables

    >>> X = np.array(db.by_col("INC"))
    >>> X = np.reshape(X, (49,1))
    >>> yd = np.array(db.by_col("HOVAL"))
    >>> yd = np.reshape(yd, (49,1))
    >>> q = np.array(db.by_col("DISCBD"))
    >>> q = np.reshape(q, (49,1))
    >>> reg = Spmodel(name_ds='columbus', w_list=[w, w2], wk_list=[], y=y, name_y='crime', x=X, name_x=['inc'],\
        ye=yd, name_ye=['hoval'], h=q, name_h=['discbd'],\
        r=None, name_r=None, s=None, name_s=None, t=None, name_t=None,\
        model_type='Standard', ols_diag=True, spat_diag=True, moran=False,\
        white_test=False, vc_matrix=True, predy_resid=False,\
        max_iter=1, stop_crit=0.00001,\
        comp_inverse='Power Expansion', step1c=False,\
        instrument_lags=1, lag_user_inst=True,\
        sig2n_k_ols=True, sig2n_k_tsls=False, sig2n_k_gmlag=False,\
        regime_err_sep=None, regime_lag_sep=None, cores=None,\
        white=False, hac=False, kp_het=False, inf_lambda=False, method='gm',\
        ml_diag=False,ml_method='full',ml_epsilon=0.00001)
    >>> print reg.output[1].name_z
    ['CONSTANT', 'inc', 'hoval']
    >>> reg = Spmodel(name_ds='columbus', w_list=[w, w2], wk_list=[], y=y, name_y='crime', x=X, name_x=['inc'],\
        ye=yd, name_ye=['hoval'], h=q, name_h=['discbd'],\
        r=None, name_r=None, s=None, name_s=None, t=None, name_t=None,\
        model_type='Standard', ols_diag=True, spat_diag=False, moran=False,\
        white_test=False, vc_matrix=False, predy_resid=False,\
        max_iter=1, stop_crit=0.00001,\
        comp_inverse='Power Expansion', step1c=False,\
        instrument_lags=1, lag_user_inst=True,\
        sig2n_k_ols=True, sig2n_k_tsls=False, sig2n_k_gmlag=False,\
        regime_err_sep=None, regime_lag_sep=None, cores=None,\
        white=True, hac=False, kp_het=False, inf_lambda=False, method='gm',\
        ml_diag=False,ml_method='full',ml_epsilon=0.00001)
    >>> print reg.output[1].name_z
    ['CONSTANT', 'inc', 'hoval']
    >>> reg = Spmodel(name_ds='columbus', w_list=[w, w2], wk_list=[wk], y=y, name_y='crime', x=X, name_x=['inc'],\
        ye=yd, name_ye=['hoval'], h=q, name_h=['discbd'],\
        r=None, name_r=None, s=None, name_s=None, t=None, name_t=None,\
        model_type='Standard', ols_diag=True, spat_diag=True, moran=False,\
        white_test=False, vc_matrix=True, predy_resid=False,\
        max_iter=1, stop_crit=0.00001,\
        comp_inverse='Power Expansion', step1c=False,\
        instrument_lags=1, lag_user_inst=True,\
        sig2n_k_ols=True, sig2n_k_tsls=False, sig2n_k_gmlag=False,\
        regime_err_sep=None, regime_lag_sep=None, cores=None,\
        white=False, hac=True, kp_het=False, inf_lambda=False, method='gm',\
        ml_diag=False,ml_method='full',ml_epsilon=0.00001)
    >>> print reg.output[1].name_z
    ['CONSTANT', 'inc', 'hoval']
    >>> reg = Spmodel(name_ds='columbus', w_list=[w, w2], wk_list=[], y=y, name_y='crime', x=X, name_x=['inc'],\
        ye=yd, name_ye=['hoval'], h=q, name_h=['discbd'],\
        r=None, name_r=None, s=None, name_s=None, t=None, name_t=None,\
        model_type='Spatial Lag', ols_diag=True, spat_diag=False, moran=False,\
        white_test=False, vc_matrix=False, predy_resid=False,\
        max_iter=1, stop_crit=0.00001,\
        comp_inverse='Power Expansion', step1c=False,\
        instrument_lags=1, lag_user_inst=True,\
        sig2n_k_ols=True, sig2n_k_tsls=False, sig2n_k_gmlag=False,\
        regime_err_sep=None, regime_lag_sep=None, cores=None,\
        white=False, hac=False, kp_het=False, inf_lambda=False, method='gm',\
        ml_diag=False,ml_method='full',ml_epsilon=0.00001)
    >>> print reg.output[1].name_z
    ['CONSTANT', 'inc', 'hoval', 'W_crime']
    >>> reg = Spmodel(name_ds='columbus', w_list=[w, w2], wk_list=[], y=y, name_y='crime', x=X, name_x=['inc'],\
        ye=yd, name_ye=['hoval'], h=q, name_h=['discbd'],\
        r=None, name_r=None, s=None, name_s=None, t=None, name_t=None,\
        model_type='Spatial Lag', ols_diag=True, spat_diag=False, moran=False,\
        white_test=False, vc_matrix=False, predy_resid=False,\
        max_iter=1, stop_crit=0.00001,\
        comp_inverse='Power Expansion', step1c=False,\
        instrument_lags=1, lag_user_inst=True,\
        sig2n_k_ols=True, sig2n_k_tsls=False, sig2n_k_gmlag=False,\
        regime_err_sep=None, regime_lag_sep=None, cores=None,\
        white=True, hac=False, kp_het=False, inf_lambda=False, method='gm',\
        ml_diag=False,ml_method='full',ml_epsilon=0.00001)
    >>> print reg.output[1].name_z
    ['CONSTANT', 'inc', 'hoval', 'W_crime']
    >>> reg = Spmodel(name_ds='columbus', w_list=[w, w2], wk_list=[], y=y, name_y='crime', x=X, name_x=['inc'],\
        ye=yd, name_ye=['hoval'], h=q, name_h=['discbd'],\
        r=None, name_r=None, s=None, name_s=None, t=None, name_t=None,\
        model_type='Spatial Lag', ols_diag=True, spat_diag=False, moran=False,\
        white_test=False, vc_matrix=False, predy_resid=False,\
        max_iter=1, stop_crit=0.00001,\
        comp_inverse='Power Expansion', step1c=False,\
        instrument_lags=1, lag_user_inst=True,\
        sig2n_k_ols=True, sig2n_k_tsls=False, sig2n_k_gmlag=False,\
        regime_err_sep=None, regime_lag_sep=None, cores=None,\
        white=False, hac=False, kp_het=False, inf_lambda=False, method='gm',\
        ml_diag=False,ml_method='full',ml_epsilon=0.00001)
    >>> print reg.output[1].name_z
    ['CONSTANT', 'inc', 'hoval', 'W_crime']
    >>> reg = Spmodel(name_ds='columbus', w_list=[w, w2], wk_list=[], y=y, name_y='crime', x=X, name_x=['inc'],\
        ye=yd, name_ye=['hoval'], h=q, name_h=['discbd'],\
        r=None, name_r=None, s=None, name_s=None, t=None, name_t=None,\
        model_type='Spatial Error', ols_diag=True, spat_diag=True, moran=False,\
        white_test=False, vc_matrix=True, predy_resid=False,\
        max_iter=1, stop_crit=0.00001,\
        comp_inverse='Power Expansion', step1c=False,\
        instrument_lags=1, lag_user_inst=True,\
        sig2n_k_ols=True, sig2n_k_tsls=False, sig2n_k_gmlag=False,\
        regime_err_sep=None, regime_lag_sep=None, cores=None,\
        white=False, hac=False, kp_het=False, inf_lambda=False, method='gm',\
        ml_diag=False,ml_method='full',ml_epsilon=0.00001)
    >>> print reg.output[1].name_z
    ['CONSTANT', 'inc', 'hoval', 'lambda']
    >>> reg = Spmodel(name_ds='columbus', w_list=[w, w2], wk_list=[], y=y, name_y='crime', x=X, name_x=['inc'],\
        ye=yd, name_ye=['hoval'], h=q, name_h=['discbd'],\
        r=None, name_r=None, s=None, name_s=None, t=None, name_t=None,\
        model_type='Spatial Error', ols_diag=True, spat_diag=True, moran=False,\
        white_test=False, vc_matrix=False, predy_resid=False,\
        max_iter=1, stop_crit=0.00001,\
        comp_inverse='Power Expansion', step1c=False,\
        instrument_lags=1, lag_user_inst=True,\
        sig2n_k_ols=True, sig2n_k_tsls=False, sig2n_k_gmlag=False,\
        regime_err_sep=None, regime_lag_sep=None, cores=None,\
        white=False, hac=False, kp_het=True, inf_lambda=False, method='gm',\
        ml_diag=False,ml_method='full',ml_epsilon=0.00001)
    >>> print reg.output[1].name_z
    ['CONSTANT', 'inc', 'hoval', 'lambda']
    >>> reg = Spmodel(name_ds='columbus', w_list=[w, w2], wk_list=[], y=y, name_y='crime', x=X, name_x=['inc'],\
        ye=yd, name_ye=['hoval'], h=q, name_h=['discbd'],\
        r=None, name_r=None, s=None, name_s=None, t=None, name_t=None,\
        model_type='Spatial Lag+Error', ols_diag=True, spat_diag=False, moran=False,\
        white_test=False, vc_matrix=False, predy_resid=False,\
        max_iter=1, stop_crit=0.00001,\
        comp_inverse='Power Expansion', step1c=False,\
        instrument_lags=1, lag_user_inst=True,\
        sig2n_k_ols=True, sig2n_k_tsls=False, sig2n_k_gmlag=False,\
        regime_err_sep=None, regime_lag_sep=None, cores=None,\
        white=False, hac=False, kp_het=False, inf_lambda=False, method='gm',\
        ml_diag=False,ml_method='full',ml_epsilon=0.00001)
    >>> print reg.output[1].name_z
    ['CONSTANT', 'inc', 'hoval', 'W_crime', 'lambda']
    >>> reg = Spmodel(name_ds='columbus', w_list=[w, w2], wk_list=[], y=y, name_y='crime', x=X, name_x=['inc'],\
        ye=yd, name_ye=['hoval'], h=q, name_h=['discbd'],\
        r=None, name_r=None, s=None, name_s=None, t=None, name_t=None,\
        model_type='Spatial Lag+Error', ols_diag=True, spat_diag=False, moran=False,\
        white_test=False, vc_matrix=True, predy_resid=False,\
        max_iter=1, stop_crit=0.00001,\
        comp_inverse='Power Expansion', step1c=False,\
        instrument_lags=1, lag_user_inst=True,\
        sig2n_k_ols=True, sig2n_k_tsls=False, sig2n_k_gmlag=False,\
        regime_err_sep=None, regime_lag_sep=None, cores=None,\
        white=False, hac=False, kp_het=True, inf_lambda=False, method='gm',\
        ml_diag=False,ml_method='full',ml_epsilon=0.00001)
    >>> print reg.output[1].name_z
    ['CONSTANT', 'inc', 'hoval', 'W_crime', 'lambda']

    """
    def __init__(
        self, name_ds, w_list, wk_list, y, name_y, x, name_x, ye, name_ye,
        h, name_h, r, name_r, s, name_s, t, name_t,
        model_type,
        spat_diag,
        white, hac, kp_het,
        sig2n_k_ols, sig2n_k_tsls, sig2n_k_gmlag,
        max_iter, stop_crit, inf_lambda, comp_inverse, step1c,
        instrument_lags, lag_user_inst,
        vc_matrix, predy_resid,
        ols_diag, moran, white_test,
        regime_err_sep, regime_lag_sep, cores, method, 
        ml_epsilon, ml_method, ml_diag, ids=None):
        
        self.name_ds = name_ds
        self.w_list = w_list
        self.wk_list = wk_list
        self.y = y
        self.name_y = name_y
        self.x = x
        self.name_x = name_x
        self.ye = ye
        self.name_ye = name_ye
        self.h = h
        self.name_h = name_h
        self.r = r
        self.name_r = name_r
        self.s = s
        self.name_s = name_s
        self.t = t
        self.name_t = name_t
        self.model_type = model_type
        self.spat_diag = spat_diag
        self.white = white
        self.hac = hac
        self.kp_het = kp_het
        self.sig2n_k_ols = sig2n_k_ols
        self.sig2n_k_tsls = sig2n_k_tsls
        self.sig2n_k_gmlag = sig2n_k_gmlag
        self.max_iter = max_iter
        self.stop_crit = stop_crit
        self.inf_lambda = inf_lambda
        self.comp_inverse = comp_inverse
        self.step1c = step1c
        self.instrument_lags = instrument_lags
        self.lag_user_inst = lag_user_inst
        self.vc_matrix = vc_matrix
        self.predy_resid = predy_resid
        self.ols_diag = ols_diag
        self.moran = moran
        self.white_test = white_test
        self.regime_err_sep = regime_err_sep
        self.regime_lag_sep = regime_lag_sep
        self.cores = cores
        self.method = method
        self.ml_epsilon = ml_epsilon
        self.ml_method = ml_method.lower()
        self.ml_diag = ml_diag
        self.ids = ids

        if predy_resid:
            self.pred_res = y.reshape(y.shape[0], 1)
            self.header_pr = name_y

        if comp_inverse == 'Power Expansion':
            self.comp_inverse = 'power_exp'
        elif comp_inverse == 'True Inverse':
            self.comp_inverse = 'true_inv'

        if name_ye:
            endog = True
        else:
            endog = False

        if name_r:
            regi = True
        else:
            regi = False

        self.output = model_getter[(
            model_type, endog, inf_lambda, regi, method)](self)

        if predy_resid:
            if not self.ids:
                self.ids = get_ids(self.w_list, self.y.shape[0])
            f = open(predy_resid, 'w')
            f.write('IDs,'+self.header_pr)
            content = ''
            for i,line in enumerate(self.pred_res):
                content += '\n'+str(self.ids[i])+',' 
                for item in line:
                    content +=  '%.4f,' %item
            f.write(content + '\n')
            f.close()

            """
            outfile = open(predy_resid, 'w')
            outfile.write(self.header_pr+'\n')
            np.savetxt(outfile, self.pred_res, delimiter=',')
            """

def spmodel(name_ds, w_list, wk_list, y, name_y, x, name_x, ye, name_ye,
            h, name_h, r, name_r, s, name_s, t, name_t,
            model_type, spat_diag, white, hac, kp_het,
            sig2n_k_ols, sig2n_k_tsls, sig2n_k_gmlag,
            max_iter, stop_crit, inf_lambda, comp_inverse, step1c,
            instrument_lags, lag_user_inst,
            vc_matrix, predy_resid,
            ols_diag, moran, white_test,
            regime_err_sep, regime_lag_sep, cores, method, ids=None):
    """
    DEPRECATED
    spmodel originally ran the dispatcher. The class Spmodel now runs the
    dispatcher, and was created when this module was refactored for ease of
    maintenance.  The spmodel function is simply a little glue so that the old
    GUI code did not need to change to accommodate the refactored dispatcher.
    """
    result = Spmodel(
        name_ds, w_list, wk_list, y, name_y, x, name_x, ye, name_ye,
        h, name_h, r, name_r, s, name_s, t, name_t,
        model_type, spat_diag, white, hac, kp_het,
        sig2n_k_ols, sig2n_k_tsls, sig2n_k_gmlag,
        max_iter, stop_crit, inf_lambda, comp_inverse, step1c,
        instrument_lags, lag_user_inst,
        vc_matrix, predy_resid,
        ols_diag, moran, white_test,
        regime_err_sep, regime_lag_sep, cores, method, ids)
    return result.output

##############################################################################
############### Helper functions for launching econometric models ############
##############################################################################


class Wildcard_Dict(dict):
    """Modified dictionary that allows for wildcards. The key is assumed to be
    a tuple. If '*' appears in the key tuple, then two keys are created where
    the '*' is replaced by True in one and False in the other; both items have
    the same value.

    Example
    -------
    >>> d = Wildcard_Dict()
    >>> d[('rrrr', True)] = 'Test Value 1'
    >>> len(d)
    1
    >>> d[('tttt', '*')] = 'Test Value 2'
    >>> len(d)
    3
    >>> d[('rrrr', True)]
    'Test Value 1'
    >>> d[('tttt', True)]
    'Test Value 2'
    >>> d[('tttt', False)]
    'Test Value 2'
    """

    def __init__(self):
        dict.__init__(self)

    def __setitem__(self, key, value):
        stars = key.count('*')
        if stars:
            new_keys = [list(COPY.copy(key)) for i in range(2**stars)]
            for index, i in enumerate(key):
                if i == '*':
                    for keys in range(0, 2**stars, 2):
                        new_keys[keys][index] = True
                        new_keys[keys+1][index] = False
            for new_key in new_keys:
                dict.__setitem__(self, tuple(new_key), value)
        else:
            dict.__setitem__(self, key, value)


def get_robust(reg_robust, robust, gwk=None, sig2n_k=False):
    """Creates a new regression object, computes the robust standard errors,
    resets the regression object's internal cache and recompute the non-spatial
    diagnostics.
    """
    reg_robust.robust = robust
    if gwk != None:
        reg_robust.name_gwk = gwk.name
    reg_robust.vm = ROBUST.robust_vm(reg=reg_robust, gwk=gwk, sig2n_k=sig2n_k)
    return reg_robust


def collect_predy_resid(pred_res, header_pr, reg, model, spatial, ws, counter):
    if ws > 1:
        lead = model+'W'+str(counter)+'_'
        counter += 1
    else:
        lead = model
    if spatial:
        pred_res = np.hstack((
            pred_res, reg.predy, reg.u, reg.predy_sp, reg.resid_sp))
        header_pr += ','+lead+'predy,'+lead + \
            'resid,'+lead+'predy_sp,'+lead+'resid_sp'
    else:
        pred_res = np.hstack((pred_res, reg.predy, reg.u))
        header_pr += ','+lead+'predy,'+lead+'resid'
    return pred_res, header_pr, counter


def get_white_hac_standard(reg, gui):
    """Test if user requested White and/or HAC standard errors. If yes then
    compute the new standard errors using the information already computed in
    the initial regression run. Standard models (OLS, TSLS).
    """
    robust_regs = []
    multireg = False
    if hasattr(reg, 'multi'):
        multireg = True
    if gui.white:
        # compute White std errors
        if len(gui.h) > 0:
            sig2n_k = gui.sig2n_k_tsls
        else:
            sig2n_k = gui.sig2n_k_ols
        if multireg:
            reg_robust = COPY.deepcopy(reg)
            reg_robust._cache = {}
            for m in reg_robust.multi:
                reg_robust.multi[m] = get_robust(reg_robust.multi[m],'white',sig2n_k=sig2n_k)
            robust_regs.append(robust_vm_multi(reg_robust))
        else:
            reg_robust = COPY.copy(reg)
            reg_robust._cache = {}
            robust_regs.append(get_robust(reg_robust,'white',sig2n_k=sig2n_k))
    if gui.hac:
        if len(gui.wk_list) == 0:
            raise Exception("must provide kernel weights matrix to use HAC")
        for gwk in gui.wk_list:
            # compute HAC std errors
            USER.check_robust('hac', gwk)
            if multireg:
                reg_robust = COPY.deepcopy(reg)
                reg_robust._cache = {}
                reg_robust.name_gwk = gwk.name
                ROBUST.hac_multi(reg_robust, gwk)
                robust_regs.append(reg_robust)
            else:
                reg_robust = COPY.copy(reg)
                reg_robust._cache = {}
                robust_regs.append(get_robust(reg_robust, 'hac', gwk))

    return robust_regs


def get_white_hac_lag(reg, gui, output):
    """Test if user requested White and/or HAC standard errors. If yes then
    compute the new standard errors using the information already computed in
    the initial regression run. GM_Lag models.
    """
    robust_regs = []
    multireg = False
    if hasattr(reg, 'multi'):
        multireg = True
    if gui.white:
        for reg in output:
            # compute White std errors
            if multireg:
                reg_robust = COPY.deepcopy(reg)
                reg_robust._cache = {}
                for m in reg_robust.multi:
                    reg_robust.multi[m] = get_robust(
                        reg_robust.multi[m], 'white',sig2n_k=gui.sig2n_k_gmlag)
                robust_regs.append(robust_vm_multi(reg_robust))
            else:
                reg_robust = COPY.copy(reg)
                reg_robust._cache = {}
                robust_regs.append(get_robust(reg_robust, 'white',sig2n_k=gui.sig2n_k_gmlag))
    if gui.hac:
        if len(gui.wk_list) == 0:
            raise Exception("must provide kernel weights matrix to use HAC")
        for reg in output:
            for gwk in gui.wk_list:
                # compute HAC std errors
                USER.check_robust('hac', gwk)
                if multireg:
                    reg_robust = COPY.deepcopy(reg)
                    reg_robust._cache = {}
                    reg_robust.name_gwk = gwk.name
                    ROBUST.hac_multi(reg_robust, gwk, constant=True)
                    robust_regs.append(reg_robust)
                else:
                    reg_robust = COPY.copy(reg)
                    reg_robust._cache = {}
                    robust_regs.append(get_robust(reg_robust, 'hac', gwk))
    return robust_regs


def robust_vm_multi(reg):
    counter = 0
    for r in reg.regimes_set:
        reg.vm[(counter*reg.kr):((counter+1)*reg.kr), (
            counter*reg.kr):((counter+1)*reg.kr)] = reg.multi[r].vm
        counter += 1
    return reg

def get_ids(w_list, N):
    try:
        if len(w_list[0].id_order)<N:
            idw = w_list[0].id_order * N/w_list[0].n
        else:
            ids = w_list[0].id_order
    except:
        ids = range(1,N+1)
    return ids

def collect_predy_resid(pred_res, header_pr, reg, model, spatial, ws, counter):
    if ws > 1:
        lead = model+'W'+str(counter)+'_'
        counter += 1
    else:
        lead = model
    if spatial:
        pred_res = np.hstack((
            pred_res, reg.predy, reg.u, reg.predy_e, reg.e_pred))
        header_pr += ','+lead+'predy,'+lead + \
            'resid,'+lead+'predy_e,'+lead+'e_pred'
    else:
        pred_res = np.hstack((pred_res, reg.predy, reg.u))
        header_pr += ','+lead+'predy,'+lead+'resid'
    return pred_res, header_pr, counter


def run_predy_resid(gui, reg, model, spatial, counter):
    if gui.predy_resid:  # write out predicted values and residuals
        gui.pred_res, gui.header_pr, counter = collect_predy_resid(
            gui.pred_res, gui.header_pr, reg, model,
            spatial, len(gui.w_list), counter)

##############################################################################
##############################################################################


##############################################################################
############### Main functions to run econometric models #####################
##############################################################################
"""
This section contains one function for each econometric model available in the
model_getter dictionary. Each function takes the relevant parameters passed
from the GUI and passes them on to a pysal regression class. We make every
effort to reuse previously computed results, so for example OLS will only
compute the betas once even if the user asks for robust standard errors.
"""


def get_OLS(gui):
    reg = OLS(y=gui.y, x=gui.x,
              nonspat_diag=gui.ols_diag, white_test=gui.white_test, spat_diag=False,
              vm=gui.vc_matrix, name_y=gui.name_y, name_x=gui.name_x, name_ds=gui.name_ds,
              sig2n_k=gui.sig2n_k_ols)
    if gui.predy_resid:  # write out predicted values and residuals
        gui.pred_res, gui.header_pr, counter = collect_predy_resid(
            gui.pred_res, gui.header_pr, reg, 'standard_',
            False, 0, 0)
    if gui.w_list and gui.spat_diag:
        output = []
        for w in gui.w_list:
            # add spatial diagnostics for each W
            reg_spat = COPY.copy(reg)
            reg_spat.name_w = w.name
            SUMMARY.spat_diag_ols(reg=reg_spat, w=w, moran=gui.moran)
            SUMMARY.summary(reg=reg_spat, vm=gui.vc_matrix, instruments=False,
                            nonspat_diag=gui.ols_diag, spat_diag=True)
            output.append(reg_spat)
    else:
        output = [reg]
    robust_regs = get_white_hac_standard(reg, gui)
    for rob_reg in robust_regs:
        SUMMARY.beta_diag_ols(rob_reg, rob_reg.robust)
        SUMMARY.summary(reg=rob_reg, vm=gui.vc_matrix, instruments=False,
                        nonspat_diag=gui.ols_diag, spat_diag=False)
    output.extend(robust_regs)
    return output


def get_TSLS(gui):
    reg = TSLS(y=gui.y, x=gui.x, yend=gui.ye, q=gui.h,
               spat_diag=False, vm=gui.vc_matrix,
               name_y=gui.name_y, name_x=gui.name_x, name_yend=gui.name_ye,
               name_q=gui.name_h, name_ds=gui.name_ds, sig2n_k=gui.sig2n_k_tsls)
    if gui.predy_resid:  # write out predicted values and residuals
        gui.pred_res, gui.header_pr, counter = collect_predy_resid(
            gui.pred_res, gui.header_pr, reg, 'standard_',
            False, 0, 0)
    if gui.w_list and gui.spat_diag:
        output = []
        for w in gui.w_list:
            # add spatial diagnostics for each W
            reg_spat = COPY.copy(reg)
            reg_spat.name_w = w.name
            SUMMARY.spat_diag_instruments(reg=reg_spat, w=w)
            SUMMARY.summary(reg=reg_spat, vm=gui.vc_matrix, instruments=True,
                            nonspat_diag=False, spat_diag=True)
            output.append(reg_spat)
    else:
        output = [reg]
    robust_regs = get_white_hac_standard(reg, gui)
    for rob_reg in robust_regs:
        SUMMARY.beta_diag(rob_reg, rob_reg.robust)
        SUMMARY.build_coefs_body_instruments(rob_reg)
        SUMMARY.summary(reg=rob_reg, vm=gui.vc_matrix, instruments=True,
                        nonspat_diag=False, spat_diag=gui.spat_diag)
    output.extend(robust_regs)
    return output


def get_GM_Lag_endog(gui):
    output = []
    counter = 1
    for w in gui.w_list:
        reg = GM_Lag(y=gui.y, x=gui.x, w=w, yend=gui.ye, q=gui.h,
                     vm=gui.vc_matrix, w_lags=gui.instrument_lags, lag_q=gui.lag_user_inst,
                     name_y=gui.name_y, name_x=gui.name_x, name_yend=gui.name_ye,
                     name_q=gui.name_h, name_ds=gui.name_ds, sig2n_k=gui.sig2n_k_gmlag,
                     spat_diag=gui.spat_diag, name_w=w.name)
        run_predy_resid(gui, reg, '', True, counter)
        output.append(reg)
        counter += 1
    robust_regs = get_white_hac_lag(reg, gui, output)
    for rob_reg in robust_regs:
        SUMMARY.beta_diag_lag(rob_reg, rob_reg.robust)
        SUMMARY.build_coefs_body_instruments(rob_reg)
        SUMMARY.summary(reg=rob_reg, vm=gui.vc_matrix, instruments=True,
                        nonspat_diag=False, spat_diag=gui.spat_diag)
    output.extend(robust_regs)
    return output


def get_GM_Lag_noEndog(gui):
    output = []
    counter = 1
    for w in gui.w_list:
        reg = GM_Lag(y=gui.y, x=gui.x, w=w, vm=gui.vc_matrix,
                     w_lags=gui.instrument_lags, lag_q=gui.lag_user_inst,
                     name_y=gui.name_y, name_x=gui.name_x, name_ds=gui.name_ds,
                     sig2n_k=gui.sig2n_k_gmlag, spat_diag=gui.spat_diag, name_w=w.name)
        run_predy_resid(gui, reg, '', True, counter)
        output.append(reg)
        counter += 1
    robust_regs = get_white_hac_lag(reg, gui, output)
    for rob_reg in robust_regs:
        SUMMARY.beta_diag_lag(rob_reg, rob_reg.robust)
        SUMMARY.build_coefs_body_instruments(rob_reg)
        SUMMARY.summary(reg=rob_reg, vm=gui.vc_matrix, instruments=True,
                        nonspat_diag=False, spat_diag=gui.spat_diag)
    output.extend(robust_regs)
    return output


def get_GM_Endog_Error_Hom(gui):
    output = []
    if gui.kp_het:
        output.extend(get_GM_Endog_Error_Het(gui))
    else:
        counter = 1
        for w in gui.w_list:
            reg = GM_Endog_Error_Hom(
                y=gui.y, x=gui.x, w=w, yend=gui.ye, q=gui.h,
                vm=gui.vc_matrix, max_iter=gui.max_iter, epsilon=gui.stop_crit,
                name_y=gui.name_y, name_x=gui.name_x, name_yend=gui.name_ye,
                name_q=gui.name_h, name_ds=gui.name_ds, name_w=w.name)
            run_predy_resid(gui, reg, '', False, counter)
            output.append(reg)
            counter += 1
    return output


def get_GM_Error_Hom(gui):
    output = []
    if gui.kp_het:
        output.extend(get_GM_Error_Het(gui))
    else:
        counter = 1
        for w in gui.w_list:
            reg = GM_Error_Hom(y=gui.y, x=gui.x, w=w, vm=gui.vc_matrix,
                               max_iter=gui.max_iter, epsilon=gui.stop_crit,
                               name_y=gui.name_y, name_x=gui.name_x, name_ds=gui.name_ds,
                               name_w=w.name)
            run_predy_resid(gui, reg, '', False, counter)
            output.append(reg)
            counter += 1
    return output


def get_GM_Endog_Error(gui):
    output = []
    if gui.kp_het:
        output.extend(get_GM_Endog_Error_Het(gui))
    else:
        counter = 1
        for w in gui.w_list:
            reg = GM_Endog_Error(y=gui.y, x=gui.x, w=w, yend=gui.ye, q=gui.h,
                                 vm=gui.vc_matrix,
                                 name_y=gui.name_y, name_x=gui.name_x, name_yend=gui.name_ye,
                                 name_q=gui.name_h, name_ds=gui.name_ds, name_w=w.name)
            run_predy_resid(gui, reg, '', False, counter)
            output.append(reg)
            counter += 1
    return output


def get_GM_Error(gui):
    output = []
    if gui.kp_het:
        output.extend(get_GM_Error_Het(gui))
    else:
        counter = 1
        for w in gui.w_list:
            reg = GM_Error(y=gui.y, x=gui.x, w=w, vm=gui.vc_matrix,
                           name_y=gui.name_y, name_x=gui.name_x, name_ds=gui.name_ds,
                           name_w=w.name)
            run_predy_resid(gui, reg, '', False, counter)
            output.append(reg)
            counter += 1
    return output


def get_GM_Endog_Error_Het(gui):
    output = []
    counter = 1
    for w in gui.w_list:
        reg = GM_Endog_Error_Het(y=gui.y, x=gui.x, w=w, yend=gui.ye, q=gui.h,
                                 vm=gui.vc_matrix, max_iter=gui.max_iter, epsilon=gui.stop_crit,
                                 step1c=gui.step1c, inv_method=gui.comp_inverse,
                                 name_y=gui.name_y, name_x=gui.name_x, name_yend=gui.name_ye,
                                 name_q=gui.name_h, name_ds=gui.name_ds, name_w=w.name)
        run_predy_resid(gui, reg, 'het_', False, counter)
        output.append(reg)
        counter += 1
    return output


def get_GM_Error_Het(gui):
    output = []
    counter = 1
    for w in gui.w_list:
        reg = GM_Error_Het(y=gui.y, x=gui.x, w=w, vm=gui.vc_matrix,
                           max_iter=gui.max_iter, epsilon=gui.stop_crit,
                           step1c=gui.step1c,
                           name_y=gui.name_y, name_x=gui.name_x, name_ds=gui.name_ds,
                           name_w=w.name)
        run_predy_resid(gui, reg, 'het_', False, counter)
        output.append(reg)
        counter += 1
    return output


def get_GM_Combo_Hom_endog(gui):
    output = []
    if gui.kp_het:
        output.extend(get_GM_Combo_Het_endog(gui))
    else:
        counter = 1
        for w in gui.w_list:
            reg = GM_Combo_Hom(y=gui.y, x=gui.x, w=w, yend=gui.ye, q=gui.h,
                               vm=gui.vc_matrix, w_lags=gui.instrument_lags, lag_q=gui.lag_user_inst,
                               max_iter=gui.max_iter, epsilon=gui.stop_crit,
                               name_y=gui.name_y, name_x=gui.name_x, name_yend=gui.name_ye,
                               name_q=gui.name_h, name_ds=gui.name_ds, name_w=w.name)
            run_predy_resid(gui, reg, '', True, counter)
            output.append(reg)
            counter += 1
    return output


def get_GM_Combo_Hom_noEndog(gui):
    output = []
    if gui.kp_het:
        output.extend(get_GM_Combo_Het_noEndog(gui))
    else:
        counter = 1
        for w in gui.w_list:
            reg = GM_Combo_Hom(y=gui.y, x=gui.x, w=w, vm=gui.vc_matrix,
                               w_lags=gui.instrument_lags, lag_q=gui.lag_user_inst,
                  max_iter=gui.max_iter, epsilon=gui.stop_crit,
                  name_y=gui.name_y, name_x=gui.name_x, name_ds=gui.name_ds,
                  name_w=w.name)
            run_predy_resid(gui, reg, '', True, counter)
            output.append(reg)
            counter += 1
    return output


def get_GM_Combo_endog(gui):
    output = []
    if gui.kp_het:
        output.extend(get_GM_Combo_Het_endog(gui))
    else:
        counter = 1
        for w in gui.w_list:
            reg = GM_Combo(y=gui.y, x=gui.x, w=w, yend=gui.ye, q=gui.h,
                  vm=gui.vc_matrix, w_lags=gui.instrument_lags, lag_q=gui.lag_user_inst,
                  name_y=gui.name_y, name_x=gui.name_x, name_yend=gui.name_ye,
                  name_q=gui.name_h, name_ds=gui.name_ds, name_w=w.name)
            run_predy_resid(gui, reg, '', True, counter)
            output.append(reg)
            counter += 1
    return output


def get_GM_Combo_noEndog(gui):
    output = []
    if gui.kp_het:
        output.extend(get_GM_Combo_Het_noEndog(gui))
    else:
        counter = 1
        for w in gui.w_list:
            reg = GM_Combo(y=gui.y, x=gui.x, w=w, vm=gui.vc_matrix,
                  w_lags=gui.instrument_lags, lag_q=gui.lag_user_inst,
                  name_y=gui.name_y, name_x=gui.name_x, name_ds=gui.name_ds,
                  name_w=w.name)
            run_predy_resid(gui, reg, '', True, counter)
            output.append(reg)
            counter += 1
    return output


def get_GM_Combo_Het_endog(gui):
    output = []
    counter = 1
    for w in gui.w_list:
        reg = GM_Combo_Het(y=gui.y, x=gui.x, w=w, yend=gui.ye, q=gui.h,
              vm=gui.vc_matrix, w_lags=gui.instrument_lags, lag_q=gui.lag_user_inst,
              max_iter=gui.max_iter, epsilon=gui.stop_crit,
              step1c=gui.step1c, inv_method=gui.comp_inverse,
              name_y=gui.name_y, name_x=gui.name_x, name_yend=gui.name_ye,
              name_q=gui.name_h, name_ds=gui.name_ds, name_w=w.name)
        run_predy_resid(gui, reg, 'het_', True, counter)
        output.append(reg)
        counter += 1
    return output


def get_GM_Combo_Het_noEndog(gui):
    output = []
    counter = 1
    for w in gui.w_list:
        reg = GM_Combo_Het(y=gui.y, x=gui.x, w=w, vm=gui.vc_matrix,
              w_lags=gui.instrument_lags, lag_q=gui.lag_user_inst,
              max_iter=gui.max_iter, epsilon=gui.stop_crit,
              step1c=gui.step1c, inv_method=gui.comp_inverse,
              name_y=gui.name_y, name_x=gui.name_x, name_ds=gui.name_ds,
              name_w=w.name)
        run_predy_resid(gui, reg, 'het_', True, counter)
        output.append(reg)
        counter += 1
    return output


def get_OLS_regimes(gui):
    if gui.w_list:
        w0 = gui.w_list[0]
        name_w0 = gui.w_list[0].name
    else:
        w0, name_w0 = None, None
    reg = OLS_Regimes(
        y=gui.y, x=gui.x, regimes=gui.r, name_regimes=gui.name_r, w=w0,
               nonspat_diag=gui.ols_diag, white_test=gui.white_test, spat_diag=gui.spat_diag, vm=gui.vc_matrix,
               name_y=gui.name_y, name_x=gui.name_x, name_ds=gui.name_ds, cores=gui.cores,
               sig2n_k=gui.sig2n_k_ols, regime_err_sep=gui.regime_err_sep, name_w=name_w0)
    if gui.predy_resid:  # write out predicted values and residuals
        gui.pred_res, gui.header_pr, counter = collect_predy_resid(
                               gui.pred_res, gui.header_pr, reg, 'standard_',
                               False, 0, 0)
    output = [reg]
    if gui.w_list and gui.spat_diag:
        if gui.regime_err_sep:  # Run regression again to account for different W.
            for w in gui.w_list[1:]:
                reg_spat = OLS_Regimes(
                    y=gui.y, x=gui.x, regimes=gui.r, name_regimes=gui.name_r, w=w,
                       nonspat_diag=gui.ols_diag, white_test=gui.white_test, spat_diag=True, vm=gui.vc_matrix,
                       name_y=gui.name_y, name_x=gui.name_x, name_ds=gui.name_ds, cores=gui.cores,
                       sig2n_k=gui.sig2n_k_ols, regime_err_sep=gui.regime_err_sep, name_w=w.name)
                output.append(reg_spat)
        else:  # Re-use results.
            for w in gui.w_list[1:]:
                # add spatial diagnostics for each W
                reg_spat = COPY.copy(reg)
                reg_spat.name_w = w.name
                SUMMARY.spat_diag_ols(reg=reg_spat, w=w, moran=gui.moran)
                SUMMARY.summary(
                    reg=reg_spat, vm=gui.vc_matrix, instruments=False,
                                nonspat_diag=gui.ols_diag, spat_diag=True)
                output.append(reg_spat)
    robust_regs = get_white_hac_standard(reg, gui)
    for rob_reg in robust_regs:
        rob_reg.w, rob_reg.name_w = None, None
        try:
            for m in rob_reg.multi:
                rob_reg.multi[m].w, rob_reg.multi[m].name_w = None, None
                SUMMARY.beta_diag_ols(rob_reg.multi[
                                      m], rob_reg.multi[m].robust)
            rob_reg.chow = REGI.Chow(rob_reg)
            SUMMARY.summary_chow(rob_reg)
            SUMMARY.summary_multi(
                reg=rob_reg, multireg=rob_reg.multi, vm=gui.vc_matrix,
                    instruments=False, nonspat_diag=gui.ols_diag, spat_diag=False)
        except:
            SUMMARY.beta_diag_ols(rob_reg, rob_reg.robust)
            rob_reg.chow = REGI.Chow(rob_reg)
            SUMMARY.summary_chow(rob_reg)
            SUMMARY.summary(reg=rob_reg, vm=gui.vc_matrix, instruments=False,
                        nonspat_diag=gui.ols_diag, spat_diag=False)
    output.extend(robust_regs)
    return output


def get_TSLS_regimes(gui):
    if gui.w_list:
        w0 = gui.w_list[0]
        name_w0 = gui.w_list[0].name
    else:
        w0, name_w0 = None, None
    #if gui.hac and gui.regime_err_sep:
    #    gui.regime_err_sep = False
    reg = TSLS_Regimes(
        y=gui.y, x=gui.x, yend=gui.ye, q=gui.h, name_w=name_w0, w=w0,
                regimes=gui.r, name_regimes=gui.name_r, regime_err_sep=gui.regime_err_sep,
                spat_diag=gui.spat_diag, vm=gui.vc_matrix, cores=gui.cores,
                name_y=gui.name_y, name_x=gui.name_x, name_yend=gui.name_ye,
                name_q=gui.name_h, name_ds=gui.name_ds, sig2n_k=gui.sig2n_k_tsls)
    if gui.predy_resid:  # write out predicted values and residuals
        gui.pred_res, gui.header_pr, counter = collect_predy_resid(
                               gui.pred_res, gui.header_pr, reg, 'standard_',
                               False, 0, 0)
    output = [reg]
    if gui.w_list and gui.spat_diag:
        if gui.regime_err_sep:  # Run regression again to account for different W.
            for w in gui.w_list[1:]:
                reg_spat = TSLS_Regimes(
                    y=gui.y, x=gui.x, yend=gui.ye, q=gui.h, name_w=w.name, w=w,
                    regimes=gui.r, name_regimes=gui.name_r, regime_err_sep=gui.regime_err_sep,
                    spat_diag=gui.spat_diag, vm=gui.vc_matrix, cores=gui.cores,
                    name_y=gui.name_y, name_x=gui.name_x, name_yend=gui.name_ye,
                    name_q=gui.name_h, name_ds=gui.name_ds, sig2n_k=gui.sig2n_k_tsls)
                output.append(reg_spat)
        else:  # Re-use results.
            for w in gui.w_list[1:]:
                # add spatial diagnostics for each W
                reg_spat = COPY.copy(reg)
                reg_spat.name_w = w.name
                SUMMARY.spat_diag_instruments(reg=reg_spat, w=w)
                SUMMARY.summary(
                    reg=reg_spat, vm=gui.vc_matrix, instruments=True,
                                nonspat_diag=False, spat_diag=True)
                output.append(reg_spat)
    robust_regs = get_white_hac_standard(reg, gui)
    for rob_reg in robust_regs:
        rob_reg.w, rob_reg.name_w = None, None
        try:
            for m in rob_reg.multi:
                rob_reg.multi[m].w, rob_reg.multi[m].name_w = None, None
                SUMMARY.beta_diag(rob_reg.multi[m], rob_reg.multi[m].robust)
                SUMMARY.build_coefs_body_instruments(rob_reg.multi[m])
            rob_reg.chow = REGI.Chow(rob_reg)
            SUMMARY.summary_chow(rob_reg)
            SUMMARY.summary_multi(
                reg=rob_reg, multireg=rob_reg.multi, vm=gui.vc_matrix,
                    instruments=True, nonspat_diag=False, spat_diag=False)
        except:
            SUMMARY.beta_diag(rob_reg, rob_reg.robust)
            SUMMARY.build_coefs_body_instruments(rob_reg)
            rob_reg.chow = REGI.Chow(rob_reg)
            SUMMARY.summary_chow(rob_reg)
            SUMMARY.summary(reg=rob_reg, vm=gui.vc_matrix, instruments=True,
                            nonspat_diag=False, spat_diag=False)
    output.extend(robust_regs)
    return output


def get_GM_Lag_endog_regimes(gui):
    output = []
    counter = 1
    #if gui.hac and gui.regime_err_sep:
    #    gui.regime_err_sep = False
    #    gui.regime_lag_sep = False
    for w in gui.w_list:
        reg = GM_Lag_Regimes(y=gui.y, x=gui.x, w=w, yend=gui.ye, q=gui.h,
              regimes=gui.r, name_regimes=gui.name_r, cores=gui.cores,
              regime_err_sep=gui.regime_err_sep, regime_lag_sep=gui.regime_lag_sep,
              vm=gui.vc_matrix, w_lags=gui.instrument_lags, lag_q=gui.lag_user_inst,
              name_y=gui.name_y, name_x=gui.name_x, name_yend=gui.name_ye,
              name_q=gui.name_h, name_ds=gui.name_ds, sig2n_k=gui.sig2n_k_gmlag,
              spat_diag=gui.spat_diag, name_w=w.name)
        run_predy_resid(gui, reg, '', True, counter)
        output.append(reg)
        counter += 1
    robust_regs = get_white_hac_lag(reg, gui, output)
    for rob_reg in robust_regs:
        try:
            for m in rob_reg.multi:
                SUMMARY.beta_diag_lag(rob_reg.multi[
                                      m], rob_reg.multi[m].robust)
                SUMMARY.build_coefs_body_instruments(rob_reg.multi[m])
            rob_reg.chow = REGI.Chow(rob_reg)
            SUMMARY.summary_chow(rob_reg)
            SUMMARY.summary_multi(
                reg=rob_reg, multireg=rob_reg.multi, vm=gui.vc_matrix,
                    instruments=True, nonspat_diag=False, spat_diag=gui.spat_diag)
        except:
            SUMMARY.beta_diag_lag(rob_reg, rob_reg.robust)
            SUMMARY.build_coefs_body_instruments(rob_reg)
            rob_reg.chow = REGI.Chow(rob_reg)
            SUMMARY.summary_chow(rob_reg)
            SUMMARY.summary(reg=rob_reg, vm=gui.vc_matrix, instruments=True,
                            nonspat_diag=False, spat_diag=gui.spat_diag)
    output.extend(robust_regs)
    return output


def get_GM_Lag_noEndog_regimes(gui):
    output = []
    counter = 1
    #if gui.hac and gui.regime_err_sep:
    #    gui.regime_err_sep = False
    #    gui.regime_lag_sep = False
    for w in gui.w_list:
        reg = GM_Lag_Regimes(y=gui.y, x=gui.x, w=w, vm=gui.vc_matrix,
              regimes=gui.r, name_regimes=gui.name_r, cores=gui.cores,
              regime_err_sep=gui.regime_err_sep, regime_lag_sep=gui.regime_lag_sep,
              w_lags=gui.instrument_lags, lag_q=gui.lag_user_inst,
              name_y=gui.name_y, name_x=gui.name_x, name_ds=gui.name_ds,
              sig2n_k=gui.sig2n_k_gmlag, spat_diag=gui.spat_diag, name_w=w.name)
        run_predy_resid(gui, reg, '', True, counter)
        output.append(reg)
        counter += 1
    robust_regs = get_white_hac_lag(reg, gui, output)
    for rob_reg in robust_regs:
        try:
            for m in rob_reg.multi:
                SUMMARY.beta_diag_lag(rob_reg.multi[
                                      m], rob_reg.multi[m].robust)
                SUMMARY.build_coefs_body_instruments(rob_reg.multi[m])
            rob_reg.chow = REGI.Chow(rob_reg)
            SUMMARY.summary_chow(rob_reg)
            SUMMARY.summary_multi(
                reg=rob_reg, multireg=rob_reg.multi, vm=gui.vc_matrix,
                    instruments=True, nonspat_diag=False, spat_diag=gui.spat_diag)
        except:
            SUMMARY.beta_diag_lag(rob_reg, rob_reg.robust)
            SUMMARY.build_coefs_body_instruments(rob_reg)
            rob_reg.chow = REGI.Chow(rob_reg)
            SUMMARY.summary_chow(rob_reg)
            SUMMARY.summary(reg=rob_reg, vm=gui.vc_matrix, instruments=True,
                            nonspat_diag=False, spat_diag=gui.spat_diag)
    output.extend(robust_regs)
    return output


def get_GM_Endog_Error_Hom_regimes(gui):
    output = []
    if gui.kp_het:
        output.extend(get_GM_Endog_Error_Het_regimes(gui))
    else:
        counter = 1
        for w in gui.w_list:
            reg = GM_Endog_Error_Hom_Regimes(
                y=gui.y, x=gui.x, w=w, yend=gui.ye, q=gui.h,
                  regimes=gui.r, name_regimes=gui.name_r,
                  regime_err_sep=gui.regime_err_sep, cores=gui.cores,
                  vm=gui.vc_matrix, max_iter=gui.max_iter, epsilon=gui.stop_crit,
                  name_y=gui.name_y, name_x=gui.name_x, name_yend=gui.name_ye,
                  name_q=gui.name_h, name_ds=gui.name_ds, name_w=w.name)
            run_predy_resid(gui, reg, '', False, counter)
            output.append(reg)
            counter += 1
    return output


def get_GM_Error_Hom_regimes(gui):
    output = []
    if gui.kp_het:
        output.extend(get_GM_Error_Het_regimes(gui))
    else:
        counter = 1
        for w in gui.w_list:
            reg = GM_Error_Hom_Regimes(y=gui.y, x=gui.x, w=w, vm=gui.vc_matrix,
                  regimes=gui.r, name_regimes=gui.name_r,
                  regime_err_sep=gui.regime_err_sep, cores=gui.cores,
                  max_iter=gui.max_iter, epsilon=gui.stop_crit,
                  name_y=gui.name_y, name_x=gui.name_x, name_ds=gui.name_ds,
                  name_w=w.name)
            run_predy_resid(gui, reg, '', False, counter)
            output.append(reg)
            counter += 1
    return output


def get_GM_Endog_Error_regimes(gui):
    output = []
    if gui.kp_het:
        output.extend(get_GM_Endog_Error_Het_regimes(gui))
    else:
        counter = 1
        for w in gui.w_list:
            reg = GM_Endog_Error_Regimes(
                y=gui.y, x=gui.x, w=w, yend=gui.ye, q=gui.h,
                  regimes=gui.r, name_regimes=gui.name_r,
                  regime_err_sep=gui.regime_err_sep, cores=gui.cores,
                  vm=gui.vc_matrix,
                  name_y=gui.name_y, name_x=gui.name_x, name_yend=gui.name_ye,
                  name_q=gui.name_h, name_ds=gui.name_ds, name_w=w.name)
            run_predy_resid(gui, reg, '', False, counter)
            output.append(reg)
            counter += 1
    return output


def get_GM_Error_regimes(gui):
    output = []
    if gui.kp_het:
        output.extend(get_GM_Error_Het_regimes(gui))
    else:
        counter = 1
        for w in gui.w_list:
            reg = GM_Error_Regimes(y=gui.y, x=gui.x, w=w, vm=gui.vc_matrix,
                  regimes=gui.r, name_regimes=gui.name_r,
                  regime_err_sep=gui.regime_err_sep, cores=gui.cores,
                  name_y=gui.name_y, name_x=gui.name_x, name_ds=gui.name_ds,
                  name_w=w.name)
            run_predy_resid(gui, reg, '', False, counter)
            output.append(reg)
            counter += 1
    return output


def get_GM_Endog_Error_Het_regimes(gui):
    output = []
    counter = 1
    for w in gui.w_list:
        reg = GM_Endog_Error_Het_Regimes(
            y=gui.y, x=gui.x, w=w, yend=gui.ye, q=gui.h,
              regimes=gui.r, name_regimes=gui.name_r,
              regime_err_sep=gui.regime_err_sep, cores=gui.cores,
              vm=gui.vc_matrix, max_iter=gui.max_iter, epsilon=gui.stop_crit,
              step1c=gui.step1c, inv_method=gui.comp_inverse,
              name_y=gui.name_y, name_x=gui.name_x, name_yend=gui.name_ye,
              name_q=gui.name_h, name_ds=gui.name_ds, name_w=w.name)
        run_predy_resid(gui, reg, 'het_', False, counter)
        output.append(reg)
        counter += 1
    return output


def get_GM_Error_Het_regimes(gui):
    output = []
    counter = 1
    for w in gui.w_list:
        reg = GM_Error_Het_Regimes(y=gui.y, x=gui.x, w=w, vm=gui.vc_matrix,
              regimes=gui.r, name_regimes=gui.name_r,
              regime_err_sep=gui.regime_err_sep, cores=gui.cores,
              max_iter=gui.max_iter, epsilon=gui.stop_crit,
              step1c=gui.step1c,
              name_y=gui.name_y, name_x=gui.name_x, name_ds=gui.name_ds,
              name_w=w.name)
        run_predy_resid(gui, reg, 'het_', False, counter)
        output.append(reg)
        counter += 1
    return output


def get_GM_Combo_Hom_endog_regimes(gui):
    output = []
    if gui.kp_het:
        output.extend(get_GM_Combo_Het_endog_regimes(gui))
    else:
        counter = 1
        for w in gui.w_list:
            reg = GM_Combo_Hom_Regimes(
                y=gui.y, x=gui.x, w=w, yend=gui.ye, q=gui.h,
                  regimes=gui.r, name_regimes=gui.name_r,
                  regime_err_sep=gui.regime_err_sep, regime_lag_sep=gui.regime_lag_sep, cores=gui.cores,
                  vm=gui.vc_matrix, w_lags=gui.instrument_lags, lag_q=gui.lag_user_inst,
                  max_iter=gui.max_iter, epsilon=gui.stop_crit,
                  name_y=gui.name_y, name_x=gui.name_x, name_yend=gui.name_ye,
                  name_q=gui.name_h, name_ds=gui.name_ds, name_w=w.name)
            run_predy_resid(gui, reg, '', True, counter)
            output.append(reg)
            counter += 1
    return output


def get_GM_Combo_Hom_noEndog_regimes(gui):
    output = []
    if gui.kp_het:
        output.extend(get_GM_Combo_Het_noEndog_regimes(gui))
    else:
        counter = 1
        for w in gui.w_list:
            reg = GM_Combo_Hom_Regimes(y=gui.y, x=gui.x, w=w, vm=gui.vc_matrix,
                  regimes=gui.r, name_regimes=gui.name_r,
                  regime_err_sep=gui.regime_err_sep, regime_lag_sep=gui.regime_lag_sep, cores=gui.cores,
                  w_lags=gui.instrument_lags, lag_q=gui.lag_user_inst,
                  max_iter=gui.max_iter, epsilon=gui.stop_crit,
                  name_y=gui.name_y, name_x=gui.name_x, name_ds=gui.name_ds,
                  name_w=w.name)
            run_predy_resid(gui, reg, '', True, counter)
            output.append(reg)
            counter += 1
    return output


def get_GM_Combo_endog_regimes(gui):
    output = []
    if gui.kp_het:
        output.extend(get_GM_Combo_Het_endog_regimes(gui))
    else:
        counter = 1
        for w in gui.w_list:
            reg = GM_Combo_Regimes(y=gui.y, x=gui.x, w=w, yend=gui.ye, q=gui.h,
                  regimes=gui.r, name_regimes=gui.name_r,
                  regime_err_sep=gui.regime_err_sep, regime_lag_sep=gui.regime_lag_sep, cores=gui.cores,
                  vm=gui.vc_matrix, w_lags=gui.instrument_lags, lag_q=gui.lag_user_inst,
                  name_y=gui.name_y, name_x=gui.name_x, name_yend=gui.name_ye,
                  name_q=gui.name_h, name_ds=gui.name_ds, name_w=w.name)
            run_predy_resid(gui, reg, '', True, counter)
            output.append(reg)
            counter += 1
    return output


def get_GM_Combo_noEndog_regimes(gui):
    output = []
    if gui.kp_het:
        output.extend(get_GM_Combo_Het_noEndog_regimes(gui))
    else:
        counter = 1
        for w in gui.w_list:
            reg = GM_Combo_Regimes(y=gui.y, x=gui.x, w=w, vm=gui.vc_matrix,
                  regimes=gui.r, name_regimes=gui.name_r,
                  regime_err_sep=gui.regime_err_sep, regime_lag_sep=gui.regime_lag_sep, cores=gui.cores,
                  w_lags=gui.instrument_lags, lag_q=gui.lag_user_inst,
                  name_y=gui.name_y, name_x=gui.name_x, name_ds=gui.name_ds,
                  name_w=w.name)
            run_predy_resid(gui, reg, '', True, counter)
            output.append(reg)
            counter += 1
    return output


def get_GM_Combo_Het_endog_regimes(gui):
    output = []
    counter = 1
    for w in gui.w_list:
        reg = GM_Combo_Het_Regimes(y=gui.y, x=gui.x, w=w, yend=gui.ye, q=gui.h,
              regimes=gui.r, name_regimes=gui.name_r,
              regime_err_sep=gui.regime_err_sep, regime_lag_sep=gui.regime_lag_sep, cores=gui.cores,
              vm=gui.vc_matrix, w_lags=gui.instrument_lags, lag_q=gui.lag_user_inst,
              max_iter=gui.max_iter, epsilon=gui.stop_crit,
              step1c=gui.step1c, inv_method=gui.comp_inverse,
              name_y=gui.name_y, name_x=gui.name_x, name_yend=gui.name_ye,
              name_q=gui.name_h, name_ds=gui.name_ds, name_w=w.name)
        run_predy_resid(gui, reg, 'het_', True, counter)
        output.append(reg)
        counter += 1
    return output


def get_GM_Combo_Het_noEndog_regimes(gui):
    output = []
    counter = 1
    for w in gui.w_list:
        reg = GM_Combo_Het_Regimes(y=gui.y, x=gui.x, w=w, vm=gui.vc_matrix,
              regimes=gui.r, name_regimes=gui.name_r,
              regime_err_sep=gui.regime_err_sep, regime_lag_sep=gui.regime_lag_sep, cores=gui.cores,
              w_lags=gui.instrument_lags, lag_q=gui.lag_user_inst,
              max_iter=gui.max_iter, epsilon=gui.stop_crit,
              step1c=gui.step1c, inv_method=gui.comp_inverse,
              name_y=gui.name_y, name_x=gui.name_x, name_ds=gui.name_ds,
              name_w=w.name)
        run_predy_resid(gui, reg, 'het_', True, counter)
        output.append(reg)
        counter += 1
    return output

def get_ML_Lag(gui):
    output = []
    counter = 1
    for w in gui.w_list:
        reg = ML_Lag(y=gui.y, x=gui.x, w=w, method=gui.ml_method, epsilon=gui.ml_epsilon,
              spat_diag=gui.ml_diag, vm=gui.vc_matrix, name_y=gui.name_y,
              name_x=gui.name_x, name_ds=gui.name_ds, name_w=w.name)
        run_predy_resid(gui, reg, 'ml_', True, counter)
        output.append(reg)
        counter += 1
    return output

def get_ML_Lag_regimes(gui):
    output = []
    counter = 1
    for w in gui.w_list:
        reg = ML_Lag_Regimes(y=gui.y, x=gui.x, regimes=gui.r, w=w, method=gui.ml_method,
              epsilon=gui.ml_epsilon, regime_lag_sep=gui.regime_lag_sep, cores=gui.cores,
              spat_diag=gui.ml_diag, vm=gui.vc_matrix, name_y=gui.name_y,name_x=gui.name_x,
              name_ds=gui.name_ds, name_w=w.name, name_regimes=gui.name_r)
        run_predy_resid(gui, reg, 'ml_', True, counter)
        output.append(reg)
        counter += 1
    return output

def get_ML_Error(gui):    
    output = []
    counter = 1
    for w in gui.w_list:
        reg = ML_Error(y=gui.y, x=gui.x, w=w, method=gui.ml_method, epsilon=gui.ml_epsilon,
              spat_diag=gui.ml_diag, vm=gui.vc_matrix, name_y=gui.name_y,
              name_x=gui.name_x, name_ds=gui.name_ds, name_w=w.name)
        run_predy_resid(gui, reg, 'ml_', False, counter)
        output.append(reg)
        counter += 1
    return output

def get_ML_Error_regimes(gui):    
    output = []
    counter = 1
    for w in gui.w_list:
        reg = ML_Error_Regimes(y=gui.y, x=gui.x, regimes=gui.r, w=w, 
              method=gui.ml_method, epsilon=gui.ml_epsilon, 
              regime_err_sep=gui.regime_err_sep, cores=gui.cores,
              spat_diag=gui.ml_diag, vm=gui.vc_matrix, name_y=gui.name_y,
              name_x=gui.name_x, name_ds=gui.name_ds, name_w=w.name,
              name_regimes=gui.name_r)
        run_predy_resid(gui, reg, 'ml_', False, counter)
        output.append(reg)
        counter += 1
    return output



def get_error_msg(gui):
    raise Exception("This specfication is currently unavailable.")


##############################################################################
##############################################################################
"""Use the model_getter dictionary to link the user input to a specific model.
This replaces a series of if-then statements that previously navigated through
the various model choices. Use '*' as a boolean wildcard when you don't care
if the item is True or False. Future additions to GeoDaSpace may require
increasing the number of elements in the tuple key, this is not a problem as
long as all the keys have the same number of elements.
"""
# model_getter[(model_type, endog, inf_lambda, regimes, method)] = model
model_getter = Wildcard_Dict()
model_getter[('Standard', False, '*', False, 'ols')] = get_OLS
model_getter[('Standard', False, '*', True, 'ols')] = get_OLS_regimes
model_getter[('Standard', True, '*', False, 'gm')] = get_TSLS
model_getter[('Spatial Lag', True, '*', False, 'gm')] = get_GM_Lag_endog
model_getter[('Spatial Lag', False, '*', False, 'gm')] = get_GM_Lag_noEndog
model_getter[('Spatial Error', True, True,
              False, 'gm')] = get_GM_Endog_Error_Hom
model_getter[('Spatial Error', False, True, False, 'gm')] = get_GM_Error_Hom
model_getter[('Spatial Error', True, False, False, 'gm')] = get_GM_Endog_Error
model_getter[('Spatial Error', False, False, False, 'gm')] = get_GM_Error
model_getter[('Spatial Lag+Error', True,
              True, False, 'gm')] = get_GM_Combo_Hom_endog
model_getter[('Spatial Lag+Error', False,
              True, False, 'gm')] = get_GM_Combo_Hom_noEndog
model_getter[('Spatial Lag+Error', True,
              False, False, 'gm')] = get_GM_Combo_endog
model_getter[('Spatial Lag+Error', False,
              False, False, 'gm')] = get_GM_Combo_noEndog
model_getter[('Standard', True, '*', True, 'gm')] = get_TSLS_regimes
model_getter[('Spatial Lag', True, '*', True, 'gm')] = get_GM_Lag_endog_regimes
model_getter[('Spatial Lag', False, '*',
              True, 'gm')] = get_GM_Lag_noEndog_regimes
model_getter[('Spatial Error', True, True,
              True, 'gm')] = get_GM_Endog_Error_Hom_regimes
model_getter[('Spatial Error', False, True,
              True, 'gm')] = get_GM_Error_Hom_regimes
model_getter[('Spatial Error', True, False,
              True, 'gm')] = get_GM_Endog_Error_regimes
model_getter[('Spatial Error', False,
              False, True, 'gm')] = get_GM_Error_regimes
model_getter[('Spatial Lag+Error', True, True,
              True, 'gm')] = get_GM_Combo_Hom_endog_regimes
model_getter[('Spatial Lag+Error', False, True,
              True, 'gm')] = get_GM_Combo_Hom_noEndog_regimes
model_getter[('Spatial Lag+Error', True, False,
              True, 'gm')] = get_GM_Combo_endog_regimes
model_getter[('Spatial Lag+Error', False, False,
              True, 'gm')] = get_GM_Combo_noEndog_regimes
model_getter[('Standard', False, '*', False, 'ml')] = get_error_msg
model_getter[('Standard', False, '*', True, 'ml')] = get_error_msg
model_getter[('Spatial Lag', False, '*', False, 'ml')] = get_ML_Lag
model_getter[('Spatial Lag', False, '*', True, 'ml')] = get_ML_Lag_regimes
model_getter[('Spatial Error', False, '*', False, 'ml')] = get_ML_Error
model_getter[('Spatial Error', False, '*', True, 'ml')] = get_ML_Error_regimes
model_getter[('Spatial Lag+Error', False, '*', False, 'ml')] = get_error_msg
model_getter[('Spatial Lag+Error', False, '*', True, 'ml')] = get_error_msg


def _test():
    import doctest
    doctest.testmod()

if __name__ == '__main__':
    _test()
