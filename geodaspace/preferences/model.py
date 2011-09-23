#System
import os
import json
#3rd Party
#Local
from geodaspace import AbstractModel
from geodaspace import DEBUG

DEBUG = True

class preferencesModel(AbstractModel):
    """
    Model for GeoDaSpace Prefernese with save/load support.
    """
    DEFAULTS = {
        'sig2n_k_other':False, 'sig2n_k_ols':True, 'sig2n_k_gmlag':False, 'sig2n_k_2sls':False,
        'gmm_epsilon':1e-05, 'gmm_inferenceOnLambda':True, 'gmm_max_iter':1, 'gmm_step1c':False, 'gmm_inv_method':'Power exp',
        'instruments_lag_q':True, 'instruments_w_lags':1,
        'other_ols_diagnostics':True, 'other_numcores':1, 'other_residualMoran':False,
        'output_save_pred_residuals':False, 'output_vm_summary':False
    }
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

    def __init__(self):
        AbstractModel.__init__(self)
        self.reset()
    def reset(self):
        self._modelData.update(self.DEFAULTS)
        self.update()
    def dump(self,fp):
        """
        Serialize this preferencesModel as a JSON formatted stream to fp
        
        fp -- open file-like obj -- must support write.
        """
        json.dump(self._modelData, fp)
    def load(self,fp):
        """
        Deserialize the contents of fp into this preferencesModel

        fp -- open file-like obj -- must support read.
        """
        dict = json.load(fp)
        self._modelData.update(dict)
        self.update()
    
if __name__ == '__main__':
    m = preferencesModel()