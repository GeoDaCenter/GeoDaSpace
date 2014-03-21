"""
Automatic spatial regression
"""

__author__ = "Luc Anselin luc.anselin@asu.edu"


import ols
import twosls_sp
import error_sp_hom
import error_sp_het



def autospace(y,x,w,gwk,opvalue=0.01,combo=False,name_y=None,name_x=None,
              name_w=None,name_gwk=None,name_ds=None):
    """
    Runs automatic spatial regression using decision tree
    
    Accounts for both heteroskedasticity and spatial autocorrelation
    
    No endogenous variables
    
    Parameters
    ----------
    y            : array
                   nx1 array for dependent variable
    x            : array
                   Two dimensional array with n rows and one column for each
                   independent (exogenous) variable, excluding the constant
    w            : pysal W object
                   Spatial weights object 
    gwk          : pysal W object
                   Kernel spatial weights needed for HAC estimation. Note:
                   matrix must have ones along the main diagonal.
    opvalue      : real
                   p-value to be used in tests; default: opvalue = 0.01
    combo        : boolean
                   flag for use of combo model rather than HAC for lag-error
                   model; default: combo = False
                   
    Returns
    -------
    results      : a dictionary with
                   results['Final Model']: one of
                        No Space - Homoskedastic
                        No Space - Heteroskedastic
                        Spatial Lag - Homoskedastic
                        Spatial Lag - Heteroskedastic
                        Spatial Error - Homoskedastic
                        Spatial Error - Heteroskedastic
                        Spatial Lag with Spatial Error - HAC
                        Spatial Lag with Spatial Error - Homoskedastic
                        Spatial Lag with Spatial Error - Heteroskedastic
                        Robust Tests not Significant - Check Model
                   results['heteroskedasticity']: True or False
                   results['spatial lag']: True or False
                   results['spatial error']: True or False
                   results['regression1']: regression object with base model (OLS)
                   results['regression2']: regression object with final model
    """
    results = {}
    results['spatial error']=False
    results['spatial lag']=False
    r1 = ols.OLS(y,x,w=w,gwk=gwk,spat_diag=True,
                 name_y=name_y,name_x=name_x,
                 name_w=name_w,name_gwk=name_gwk,
                 name_ds=name_ds)
    results['regression1'] = r1
    Het = r1.koenker_bassett['pvalue']
    if Het < opvalue:
        Hetflag = True
    else:
        Hetflag = False
    results['heteroskedasticity'] = Hetflag
    LMError1 = r1.lm_error[1]
    LMLag1 = r1.lm_lag[1]
    if LMError1 < opvalue and LMLag1 < opvalue:
        RLMError1 = r1.rlm_error[1]
        RLMLag1 = r1.rlm_lag[1]
        if RLMError1 < opvalue and RLMLag1 < opvalue:
            results['spatial lag']=True
            results['spatial error']=True
            if not combo:
                r2 = twosls_sp.GM_Lag(y,x,w=w,gwk=gwk,robust='hac',name_y=name_y,
                                      name_x=name_x,name_w=name_w,name_gwk=name_gwk,
                                      name_ds=name_ds)
                results['final model']="Spatial Lag with Spatial Error - HAC"
            elif Hetflag:
                r2 = error_sp_het.GM_Combo_Het(y,x,w=w,name_y=name_y,name_x=name_x,
                                               name_w=name_w,name_ds=name_ds)
                results['final model']="Spatial Lag with Spatial Error - Heteroskedastic"
            else:
                r2 = error_sp_hom.GM_Combo_Hom(y,x,w=w,name_y=name_y,name_x=name_x,
                                               name_w=name_w,name_ds=name_ds)
                results['final model']="Spatial Lag with Spatial Error - Homoskedastic"
        elif RLMError1 < opvalue:
            results['spatial error']=True
            if Hetflag:
                r2 = error_sp_het.GM_Error_Het(y,x,w,name_y=name_y,name_x=name_x,
                                               name_w=name_w,name_ds=name_ds)
                results['final model']="Spatial Error - Heteroskedastic"
            else:
                r2 = error_sp_hom.GM_Error_Hom(y,x,w,name_y=name_y,name_x=name_x,
                                               name_w=name_w,name_ds=name_ds)
                results['final model']="Spatial Error - Homoskedastic"
        elif RLMLag1 < opvalue:
            results['spatial lag']=True
            if Hetflag:
                r2 = twosls_sp.GM_Lag(y,x,w=w,robust='white',
                                      name_y=name_y,name_x=name_x,
                                      name_w=name_w,name_ds=name_ds)
                results['final model']="Spatial Lag - Heteroskedastic"
            else:
                r2 = twosls_sp.GM_Lag(y,x,w=w,name_y=name_y,name_x=name_x,
                                      name_w=name_w,name_ds=name_ds)
                results['final model']="Spatial Lag - Homoskedastic"
        else:
            results['final model']="Robust Tests not Significant - Check Model"
            r2 = None
    elif LMError1 < opvalue:
        results['spatial error']=True
        if Hetflag:
            r2 = error_sp_het.GM_Error_Het(y,x,w,name_y=name_y,name_x=name_x,
                                           name_w=name_w,name_ds=name_ds)
            results['final model']="Spatial Error - Heteroskedastic"
        else:
            r2 = error_sp_hom.GM_Error_Hom(y,x,w,name_y=name_y,name_x=name_x,
                                           name_w=name_w,name_ds=name_ds)
            results['final model']="Spatial Error - Homoskedastic"
    elif LMLag1 < opvalue:
        results['spatial lag']=True
        if Hetflag:
            r2 = twosls_sp.GM_Lag(y,x,w=w,robust='white',
                                  name_y=name_y,name_x=name_x,
                                  name_w=name_w,name_ds=name_ds)
            results['final model']="Spatial Lag - Heteroskedastic"
        else:
            r2 = twosls_sp.GM_Lag(y,x,w=w,name_y=name_y,name_x=name_x,
                                  name_w=name_w,name_ds=name_ds)
            results['final model']="Spatial Lag - Homoskedastic"
    else:
        if Hetflag:
            r2 = ols.OLS(y,x,robust='white',name_y=name_y,name_x=name_x,
                         name_ds=name_ds)
            results['final model']="No Space - Heteroskedastic"
        else:
            r2 = r1
            results['final model']="No Space - Homoskedastic"
    results['regression2'] = r2
    return results