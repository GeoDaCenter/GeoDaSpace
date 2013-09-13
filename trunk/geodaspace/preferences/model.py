# System
import json
try:
    import multiprocessing
    CPU_COUNT = multiprocessing.cpu_count()
except:
    CPU_COUNT = 1
# 3rd Party
# Local
from geodaspace import AbstractModel
from geodaspace import DEBUG


class preferencesModel(AbstractModel):
    """
    Model for GeoDaSpace Preferences with save/load support.
    """
    DEFAULTS = {
        'sig2n_k_other': False,
        'sig2n_k_ols': True,
        'sig2n_k_gmlag': False,
        'sig2n_k_2sls': False,
        'gmm_epsilon': 1e-05,
        'gmm_inferenceOnLambda': True,
        'gmm_max_iter': 1,
        'gmm_step1c': False,
        'gmm_inv_method': 'Power Expansion',
        'instruments_lag_q': True,
        'instruments_w_lags': 1,
        'other_ols_diagnostics': True,
        'white_test': False,
        'other_numcores': CPU_COUNT,
        'other_residualMoran': False,
        'other_missingValueCheck': False,
        'other_missingValue': 0.0,
        'output_save_pred_residuals': False,
        'output_vm_summary': False,
        'output_show_detailed_spec': False,
        'regimes_regime_error': True,
        'regimes_regime_lag': False
    }
    sig2n_k_other = AbstractModel.abstractProp('sig2n_k_other', bool)
    sig2n_k_ols = AbstractModel.abstractProp('sig2n_k_ols', bool)
    sig2n_k_2sls = AbstractModel.abstractProp('sig2n_k_2sls', bool)
    sig2n_k_gmlag = AbstractModel.abstractProp('sig2n_k_gmlag', bool)
    gmm_epsilon = AbstractModel.abstractProp('gmm_epsilon', float)
    gmm_inferenceOnLambda = AbstractModel.abstractProp(
        'gmm_inferenceOnLambda', bool)
    gmm_max_iter = AbstractModel.abstractProp('gmm_max_iter', int)
    gmm_step1c = AbstractModel.abstractProp('gmm_step1c', bool)
    gmm_inv_method = AbstractModel.abstractProp('gmm_inv_method', str)
    instruments_lag_q = AbstractModel.abstractProp('instruments_lag_q', bool)
    instruments_w_lags = AbstractModel.abstractProp('instruments_w_lags', int)
    other_ols_diagnostics = AbstractModel.abstractProp(
        'other_ols_diagnostics', bool)
    white_test = AbstractModel.abstractProp(
        'white_test', bool)
    other_numcores = AbstractModel.abstractProp('other_numcores', int)
    other_residualMoran = AbstractModel.abstractProp(
        'other_residualMoran', bool)
    other_missingValueCheck = AbstractModel.abstractProp(
        'other_missingValueCheck', bool)
    other_missingValue = AbstractModel.abstractProp(
        'other_missingValue', float)
    output_save_pred_residuals = AbstractModel.abstractProp(
        'output_save_pred_residuals', bool)
    output_vm_summary = AbstractModel.abstractProp('output_vm_summary', bool)
    output_show_detailed_spec = AbstractModel.abstractProp(
        'output_show_detailed_spec', bool)
    regimes_regime_error = AbstractModel.abstractProp(
        'regimes_regime_error', bool)
    regimes_regime_lag = AbstractModel.abstractProp('regimes_regime_lag', bool)

    def __init__(self):
        AbstractModel.__init__(self)
        self.reset()

    def reset(self):
        self._modelData.update(self.DEFAULTS)
        self.update()

    def dump(self, fp):
        """
        Serialize this preferencesModel as a JSON formatted stream to fp

        fp -- open file-like obj -- must support write.
        """
        json.dump(self._modelData, fp)

    def load(self, fp):
        """
        Deserialize the contents of fp into this preferencesModel

        fp -- open file-like obj -- must support read.
        """
        dict = json.load(fp)
        self._modelData.update(dict)
        self.update()

if __name__ == '__main__':
    m = preferencesModel()
