#System
import os
#3rd Party
#Local
from geodaspace import AbstractModel
from geodaspace import DEBUG

DEBUG = True

class preferencesModel(AbstractModel):
    DEFAULTS = {
        'sig2n_k_other':False, 'sig2n_k_ols':True, 'sig2n_k_gmlag':False, 'sig2n_k_2sls':False,
        'gmm_epsilon':1e-05, 'gmm_inferenceOnLambda':True, 'gmm_max_iter':1, 'gmm_step1c':False, 'gmm_inv_method':'power_exp',
        'instruments_lag_q':True, 'instruments_w_lags':1,
        'other_ols_diagnostics':True, 'other_numcores':1, 'other_residualMoran':False,
        'output_save_pred_residuals':False, 'output_vm_summary':False
    }
    def __init__(self):
        AbstractModel.__init__(self)
        self.reset()
    def reset(self):
        self._modelData.update(self.DEFAULTS)

    """
    'sig2n_k': {'other': False, 'ols': True, 'gmlag': False, '2sls': False},
    'gmm': {'epsilon': 1e-05, 'inferenceOnLambda': True, 'max_iter': 1, 'step1c': False, 'inv_method': 'power_exp'},
    'instruments': {'lag_q': True, 'w_lags': 1},
    'other': {'ols_diagnostics': True, 'numcores': 1, 'residualMoran': False},
    'output': {'save_pred_residuals': False, 'vm_summary': False}}
    """

    sig2n_k_other = AbstractModel.abstractProp('sig2n_k_other',bool)
    sig2n_k_ols = AbstractModel.abstractProp('sig2n_k_ols',bool)
    sig2n_k_2sls = AbstractModel.abstractProp('sig2n_k_2sls',bool)
    sig2n_k_gmlag = AbstractModel.abstractProp('sig2n_k_gmlag',bool)

    gmm_epsilon  = AbstractModel.abstractProp('gmm_epsilon',float)
    gmm_inferenceOnLambda = AbstractModel.abstractProp('gmm_inferenceOnLambda',bool)
    gmm_max_iter = AbstractModel.abstractProp('gmm_max_iter',int)
    gmm_step1c = AbstractModel.abstractProp('gmm_step1c',bool)
    gmm_inv_method = AbstractModel.abstractProp('gmm_inv_method',str)

    instruments_lag_q = AbstractModel.abstractProp('instruments_lag_q',bool)
    instruments_w_lags = AbstractModel.abstractProp('instruments_w_lags',int)

    other_ols_diagnostics = AbstractModel.abstractProp('other_ols_diagnostics',bool)
    other_numcores = AbstractModel.abstractProp('other_numcores',int)
    other_residualMoran = AbstractModel.abstractProp('other_residualMoran',bool)

    output_save_pred_residuals = AbstractModel.abstractProp('output_save_pred_residuals',bool)
    output_vm_summary = AbstractModel.abstractProp('output_vm_summary',bool)
    
if __name__ == '__main__':
    m = preferencesModel()
