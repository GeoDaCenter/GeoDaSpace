# system
import os
# 3rd Part
import wx
# local
from geodaspace import remapEvtsToDispatcher, DEBUG
from model import preferencesModel
import view
from tooltips import tips

from econometrics.gs_dispatcher import INV_METHODS
from econometrics.gs_dispatcher import ML_METHODS
# INV_METHODS = ('Power exp','True inv',)
# ML_METHODS = ('Full','Ord',)

STD_DEV_PAGE = 0
GMM_PAGE = 1
INSTRUMENTS_PAGE = 2
OUTPUT_PAGE = 3
OTHER_PAGE = 4


class preferencesDialog(view.xrcgsPrefsDialog):
    """
    GeoDaSpace Preference Dialog --
    Displays a Dialog for editing GeoDaSpace Preferences

    Display using ShowModal, which will return, wx.ID_OK or wx.ID_CANCEL

    Parameters
    ----------
    parent -- wxWindow -- A parent to the dialog, optional.

    Methods
    -------
    GetPrefs -- returns a dict contain preference settings.
    SetPrefs -- dict -- Set the preference settings from the dict.

    Notes
    -----
    Settings Dictionary Format,

    Proper Usage
    ------------
    configDlg = preferencesDialog(parent_frame)
    configDlg.ShowModal()
    #if ShowModal ends with wx.ID_CANCEL, prefs are defaults.
    prefs = configDlg.GetPrefs()

    """
    # DEFAULTS = {
    # 'sig2n_k': {'other': False, 'ols': True, 'gmlag': False, '2sls': False},
    # 'gmm': {'epsilon': 1e-05, 'inferenceOnLambda': True, 'max_iter': 1,
    # 'step1c': False, 'inv_method': 'power_exp'},
    # 'instruments': {'lag_q': True, 'w_lags': 1},
    # 'other':{'ols_diagnostics': True, 'numcores': 0, 'residualMoran': False},
    # 'output': {'save_pred_residuals': False, 'vm_summary': False}}

    def __init__(self, parent=None):
        self.__mod = False
        remapEvtsToDispatcher(self, self.evtDispatch)
        view.xrcgsPrefsDialog.__init__(self, parent)
        self.CompInverse.SetItems(list(INV_METHODS))
        self.MLMethod.SetItems(list(ML_METHODS))
        #self.numcores.SetItems(map(str, CPU_OPTIONS))
        self.SetEscapeId(self.cancelButton.GetId())
        self.SetAffirmativeId(self.saveButton.GetId())

        for widget in tips:
            try:
                getattr(self, widget).SetToolTipString(tips[widget])
                getattr(self, widget + 'Label').SetToolTipString(tips[widget])
            except:
                print "could not set tool tip for %s" % widget

        self.dispatch = d = {}
        d['saveButton'] = self.save
        d['restoreButton'] = self.restore
        d['cancelButton'] = self.cancel
        d['sig2n_k_ols'] = self.sig2n_k_ols
        d['OLSNk'] = self.sig2n_k_ols
        d['OLSN'] = self.sig2n_k_ols
        d['sig2n_k_2sls'] = self.sig2n_k_2sls
        d['twoSLSNk'] = self.sig2n_k_2sls
        d['twoSLSN'] = self.sig2n_k_2sls
        d['sig2n_k_gmlag'] = self.sig2n_k_gm
        d['GMlagNk'] = self.sig2n_k_gm
        d['GMlagN'] = self.sig2n_k_gm
        d['gmm_max_iter'] = self.gmm_max_iter
        d['MaxIterations'] = self.gmm_max_iter
        d['StoppingCriterion'] = self.gmm_epsilon
        d['gmm_epsilon'] = self.gmm_epsilon
        d['inferenceOnLambda'] = self.gmm_inferenceOnLambda
        d['gmm_inferenceOnLambda'] = self.gmm_inferenceOnLambda
        d['gmm_inv_method'] = self.gmm_inv_method
        d['CompInverse'] = self.gmm_inv_method
        d['gmm_step1c'] = self.gmm_step1c
        d['Step1c'] = self.gmm_step1c
        d['NumSpatialLags'] = self.instruments_w_lags
        d['instruments_w_lags'] = self.instruments_w_lags
        d['IncludeLagsofUserInst'] = self.instruments_lag_q
        d['instruments_lag_q'] = self.instruments_lag_q
        d['output_vm_summary'] = self.output_vm_summary
        d['ShowVarCovarMatrix'] = self.output_vm_summary
        d['output_save_pred_residuals'] = self.output_save_pred_residuals
        d['saveValuesResiduals'] = self.output_save_pred_residuals
        d['showDetailedModelSpec'] = self.output_show_detailed_spec
        d['output_show_detailed_spec'] = self.output_show_detailed_spec
        d['other_ols_diagnostics'] = self.other_ols_diagnostics
        d['OLSdiagnostics'] = self.other_ols_diagnostics
        d['white_test'] = self.white_test
        d['WhiteTest'] = self.white_test
        d['other_numcores'] = self.other_numcores
        d['numcores'] = self.other_numcores
        d['other_residualMoran'] = self.other_residualMoran
        d['residualMoran'] = self.other_residualMoran
        d['missingValueCheck'] = self.other_missingValueCheck
        d['other_missingValueCheck'] = self.other_missingValueCheck
        d['missingValue'] = self.other_missingValue
        d['other_missingValue'] = self.other_missingValue
        d['regimes_regime_error'] = self.regimes_regime_error
        d['RegimeError'] = self.regimes_regime_error
        d['regimes_regime_lag'] = self.regimes_regime_lag
        d['RegimeLag'] = self.regimes_regime_lag
        d['ml_diagnostics'] = self.ml_diagnostics
        d['MLdiagnostics'] = self.ml_diagnostics
        d['ml_epsilon'] = self.ml_epsilon
        d['MLToleranceCriterion'] = self.ml_epsilon
        d['ml_method'] = self.ml_method
        d['MLMethod'] = self.ml_method

        self.model = preferencesModel()
        self.reset_model()
        self.modified = False
        self.model.addListener(self.update)
        self.update()

    def reset_model(self):
        self.model.reset()
        if os.path.exists(self.config_file):
            try:
                config_fp = open(self.config_file, 'r')
                self.model.load(config_fp)
                config_fp.close()
                self.update()
            except:
                self.model.reset()

    def update(self, tag=False):
        if DEBUG:
            print "CONTROL... updating tag:", tag
        if tag:
            if tag in self.dispatch:
                self.dispatch[tag](value=self.model.getByTag(tag))
            else:
                if DEBUG:
                    print "Warning: %s, has not been implemented" % tag
        else:
            for key, value in self.model:
                if key in self.dispatch:
                    self.dispatch[key](value=value)
                else:
                    if DEBUG:
                        print "Warning: %s, has not been implemented" % key
        if self.modified:
            self.SetTitle("GeoDaSpace Preferences*")
        else:
            self.SetTitle("GeoDaSpace Preferences")

    def evtDispatch(self, evtName, evt):
        evtName, widgetName = evtName.rsplit('_', 1)
        if widgetName not in ['restoreButton', 'saveButton', 'cancelButton']:
            self.modified = True
        if widgetName in self.dispatch:
            self.dispatch[widgetName](evtName, evt)
        else:
            if DEBUG:
                print "not implemented:", evtName, widgetName

    def OnClose(self, evt):
        return self.cancel(evt=evt)

    @property
    def config_file(self):
        paths = wx.StandardPaths_Get()
        return os.path.join(paths.GetUserConfigDir(), 'GeoDaSpace.config')

    def __set_modified(self, val):
        self.__mod = bool(val)
        if self.__mod:
            self.SetTitle("GeoDaSpace Preferences*")
        else:
            self.SetTitle("GeoDaSpace Preferences")

    def __get_modified(self):
        return self.__mod
    modified = property(fget=__get_modified, fset=__set_modified)

    def error(self, msg, tagline="An Error has occurred"):
        """ Display an error message to the user """
        dlg = wx.MessageDialog(
            self, msg, tagline, style=wx.OK | wx.ICON_ERROR).ShowModal()
        return dlg

    def cancel(self, evtName=None, evt=None, value=None):
        if self.modified:
            dlg = wx.MessageDialog(self, "Unsaved changes will be lost.",
                                   "Are you sure you wish to cancel?",
                                   style=wx.YES_NO | wx.ICON_QUESTION)
            if dlg.ShowModal() == wx.ID_YES:
                self.reset_model()
                if self.IsModal():
                    self.EndModal(wx.ID_CANCEL)
                else:
                    return wx.ID_CANCEL
            else:
                pass
        else:
            if self.IsModal():
                self.EndModal(wx.ID_CANCEL)
            else:
                return wx.ID_CANCEL

    def validate(self):
        try:
            epsilon = float(self.StoppingCriterion.GetValue())
            if epsilon >= 1.0:
                raise ValueError
        except:
            self.prefNoteBook.SetSelection(GMM_PAGE)
            self.StoppingCriterion.SetFocus()
            self.error("Stopping Criterion must be a number < 1.0")
            return False
        return True

    def save(self, evtName=None, evt=None, value=None):
        if self.validate():
            config_fp = open(self.config_file, 'w')
            self.model.dump(config_fp)
            config_fp.close()
            self.modified = False
            if self.IsModal():
                self.EndModal(wx.ID_OK)
            else:
                return wx.ID_OK

    def restore(self, evtName=None, evt=None, value=None):
        dlg = wx.MessageDialog(self, "All unsaved preferences will be lost.",
                               "Are you sure you wish to restore defaults?",
                               style=wx.CANCEL | wx.OK | wx.ICON_QUESTION)
        if dlg.ShowModal() == wx.ID_OK:
            self.model.reset()
            self.modified = True

    def sig2n_k_ols(self, evtName=None, evt=None, value=None):
        if evt:
            self.model.sig2n_k_ols = self.OLSNk.GetValue()
        elif value is not None:
            self.OLSNk.SetValue(self.model.sig2n_k_ols)
            self.OLSN.SetValue(not self.model.sig2n_k_ols)

    def sig2n_k_2sls(self, evtName=None, evt=None, value=None):
        if evt:
            self.model.sig2n_k_2sls = self.twoSLSNk.GetValue()
        elif value is not None:
            self.twoSLSNk.SetValue(self.model.sig2n_k_2sls)
            self.twoSLSN.SetValue(not self.model.sig2n_k_2sls)

    def sig2n_k_gm(self, evtName=None, evt=None, value=None):
        if evt:
            self.model.sig2n_k_gmlag = self.GMlagNk.GetValue()
        elif value is not None:
            self.GMlagNk.SetValue(self.model.sig2n_k_gmlag)
            self.GMlagN.SetValue(not self.model.sig2n_k_gmlag)

    def gmm_max_iter(self, evtName=None, evt=None, value=None):
        if evt:
            self.model.gmm_max_iter = self.MaxIterations.GetValue()
        elif value is not None:
            self.MaxIterations.SetValue(self.model.gmm_max_iter)

    def gmm_epsilon(self, evtName=None, evt=None, value=None):
        if evt:
            try:
                self.model.gmm_epsilon = float(
                    self.StoppingCriterion.GetValue())
            except:
                pass
        elif value is not None:
            try:
                curval = float(self.StoppingCriterion.GetValue())
            except:
                curval = None
            if self.model.gmm_epsilon != curval:
                self.StoppingCriterion.SetValue(str(self.model.gmm_epsilon))

    def gmm_inferenceOnLambda(self, evtName=None, evt=None, value=None):
        if evt:
            self.model.gmm_inferenceOnLambda = self.inferenceOnLambda.GetValue(
            )
        elif value is not None:
            self.inferenceOnLambda.SetValue(self.model.gmm_inferenceOnLambda)

    def gmm_inv_method(self, evtName=None, evt=None, value=None):
        if evt:
            self.model.gmm_inv_method = INV_METHODS[
                self.CompInverse.GetSelection()]
        elif value is not None:
            self.CompInverse.SetSelection(
                INV_METHODS.index(self.model.gmm_inv_method))

    def gmm_step1c(self, evtName=None, evt=None, value=None):
        if evt:
            self.model.gmm_step1c = self.Step1c.GetValue()
        elif value is not None:
            self.Step1c.SetValue(self.model.gmm_step1c)

    def instruments_w_lags(self, evtName=None, evt=None, value=None):
        if evt:
            self.model.instruments_w_lags = self.NumSpatialLags.GetValue()
        elif value is not None:
            self.NumSpatialLags.SetValue(self.model.instruments_w_lags)

    def instruments_lag_q(self, evtName=None, evt=None, value=None):
        if evt:
            self.model.instruments_lag_q = self.IncludeLagsofUserInst.GetValue(
            )
        elif value is not None:
            self.IncludeLagsofUserInst.SetValue(self.model.instruments_lag_q)

    def output_vm_summary(self, evtName=None, evt=None, value=None):
        if evt:
            self.model.output_vm_summary = self.ShowVarCovarMatrix.GetValue()
        elif value is not None:
            self.ShowVarCovarMatrix.SetValue(self.model.output_vm_summary)

    def output_save_pred_residuals(self, evtName=None, evt=None, value=None):
        if evt:
            self.model.output_save_pred_residuals = \
                self.saveValuesResiduals.GetValue()
        elif value is not None:
            self.saveValuesResiduals.SetValue(
                self.model.output_save_pred_residuals)

    def output_show_detailed_spec(self, evtName=None, evt=None, value=None):
        if evt:
            self.model.output_show_detailed_spec = \
                self.showDetailedModelSpec.GetValue()
        elif value is not None:
            self.showDetailedModelSpec.SetValue(
                self.model.output_show_detailed_spec)

    def other_ols_diagnostics(self, evtName=None, evt=None, value=None):
        if evt:
            self.model.other_ols_diagnostics = self.OLSdiagnostics.GetValue()
        elif value is not None:
            self.OLSdiagnostics.SetValue(self.model.other_ols_diagnostics)

    def white_test(self, evtName=None, evt=None, value=None):
        if evt:
            self.model.white_test = self.WhiteTest.GetValue()
        elif value is not None:
            self.WhiteTest.SetValue(self.model.white_test)

    def other_numcores(self, evtName=None, evt=None, value=None):
        if evt:
            self.model.other_numcores = self.numcores.GetValue()
        elif value is not None:
            self.numcores.SetValue(self.model.other_numcores)

    def other_residualMoran(self, evtName=None, evt=None, value=None):
        if evt:
            self.model.other_residualMoran = self.residualMoran.GetValue()
        elif value is not None:
            self.residualMoran.SetValue(self.model.other_residualMoran)

    def other_missingValueCheck(self, evtName=None, evt=None, value=None):
        if evt:
            self.model.other_missingValueCheck = \
                self.missingValueCheck.GetValue()
        elif value is not None:
            self.missingValueCheck.SetValue(self.model.other_missingValueCheck)

    def other_missingValue(self, evtName=None, evt=None, value=None):
        if evt:
            try:
                self.model.other_missingValue = float(
                    self.missingValue.GetValue())
            except:
                pass
        elif value is not None:
            try:
                curval = float(self.missingValue.GetValue())
            except:
                curval = None
            if self.model.other_missingValue != curval:
                self.missingValue.SetValue(str(self.model.other_missingValue))

    def regimes_regime_error(self, evtName=None, evt=None, value=None):
        if evt:
            self.model.regimes_regime_error = self.RegimeError.GetValue()
        elif value is not None:
            if value is False and self.model.regimes_regime_lag is True:
                self.model.regimes_regime_error = True
            else:
                self.RegimeError.SetValue(value)

    def regimes_regime_lag(self, evtName=None, evt=None, value=None):
        if evt:
            self.model.regimes_regime_lag = self.RegimeLag.GetValue()
        elif value is not None:
            self.RegimeLag.SetValue(self.model.regimes_regime_lag)
            if value:
                self.model.regimes_regime_error = True
                self.RegimeError.Disable()
            else:
                self.RegimeError.Enable()

    def ml_diagnostics(self, evtName=None, evt=None, value=None):
        if evt:
            self.model.ml_diagnostics = self.MLdiagnostics.GetValue()
        elif value is not None:
            self.MLdiagnostics.SetValue(self.model.ml_diagnostics)

    def ml_epsilon(self, evtName=None, evt=None, value=None):
        if evt:
            try:
                self.model.ml_epsilon = float(
                    self.MLToleranceCriterion.GetValue())
            except:
                pass
        elif value is not None:
            try:
                curval = float(self.MLToleranceCriterion.GetValue())
            except:
                curval = None
            if self.model.ml_epsilon != curval:
                self.MLToleranceCriterion.SetValue(str(self.model.ml_epsilon))

    def ml_method(self, evtName=None, evt=None, value=None):
        if evt:
            self.model.ml_method = ML_METHODS[
                self.MLMethod.GetSelection()]
        elif value is not None:
            self.MLMethod.SetSelection(
                ML_METHODS.index(self.model.ml_method))


    def SetPrefs(self, prefs):
        for key in prefs:
            if hasattr(self.model, key):
                self.model.setattr(key, prefs[key])
        self.model.update()

    def GetPrefs(self):
        d = {}
        d.update(self.model._modelData)
        return d
