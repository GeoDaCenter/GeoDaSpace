#system
#3rd Part
import wx
#local
from geodaspace import remapEvtsToDispatcher#,DEBUG
import preferences_xrc

DEBUG = True

STD_DEV_PAGE = 0
GMM_PAGE = 1
INSTRUMENTS_PAGE = 2
OUTPUT_PAGE = 3
OTHER_PAGE = 4
class preferencesFrame(preferences_xrc.xrcgsPrefsFrame):
    """
    GeoDaSpace Preference Dialog -- Displays a Dialog for editing GeoDaSpace Preferences

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
    {'stddev':{'ols':True,'2sls':False...
    """
    def __init__(self,parent=None):
        remapEvtsToDispatcher(self, self.evtDispatch)
        preferences_xrc.xrcgsPrefsFrame.__init__(self,parent)

        self.dispatch = d = {}
        d['saveButton'] = self.save

    def evtDispatch(self,evtName, evt):
        evtName,widgetName = evtName.rsplit('_',1)
        if widgetName in self.dispatch:
            self.dispatch[widgetName](evtName,evt)
        else:
            if DEBUG: print "not implemented:", evtName, widgetName

    def error(self, msg, tagline="An Error has occurred"):
        """ Display an error message to the user """
        wx.MessageDialog(self, msg, tagline, style=wx.OK|wx.ICON_ERROR).ShowModal()
        return
        
    def cancel(self, evtName=None, evt=None, value=None):
        print "hide dialog"
        return wx.ID_CANCEL
    def save(self, evtName=None, evt=None, value=None):
        #validation section, local names used in settings dict below.
        try:
            epsilon = float(self.StoppingCriterion.GetValue())
            if epsilon >= 1.0:
                raise ValueError
        except:
            self.prefNoteBook.SetSelection(GMM_PAGE)
            self.StoppingCriterion.SetFocus()
            self.error("Stopping Criterion must be a number < 1.0")
            return

        print evtName, evt, value
        print "OLS  stddev.:", self.OLSNk.GetValue(), self.OLSN.GetValue()
        print "2SLS stddev.:", self.twoSLSNk.GetValue(), self.twoSLSN.GetValue()
        print "GMlg stddev.:", self.GMlagNk.GetValue(), self.GMlagN.GetValue()
        print "Othr stddev.:", False, self.othermodelsN.GetValue()
        s = {}
        s['sig2n_k'] = { 'ols':self.OLSNk.GetValue(),
                        '2sls':self.twoSLSNk.GetValue(),
                        'gmlag':self.GMlagNk.GetValue(),
                        'other':(not self.othermodelsN.GetValue()) }
        s['gmm'] = {'max_iter':self.MaxIterations.GetValue(),
                    'epsilon': epsilon,
                    'inferenceOnLambda': self.inferenceOnLambda.GetValue(),
                    'inv_method': {0:'power_exp',1:'true_inverse'}[self.CompInverse.GetSelection()],
                    'step1c': self.Step1c.GetValue()
                   }
        s['instruments'] = {'w_lags':self.NumSpatialLags.GetValue(), 'lag_q':self.IncludeLagsofUserInst.GetValue()}
        s['output'] = {'vm_summary':self.ShowVarCovarMatrix.GetValue(), 'save_pred_residuals':saelf.saveValuesResiduals.GetValue()}
        s['other'] = {'ols_diagnostics':self.OLSdiagnostics.GetValue(), 'residualMoran',self.residualMoran.GetValue()}
        self._prefs = s
        print "hide dialog"
        return wx.ID_OK
            

