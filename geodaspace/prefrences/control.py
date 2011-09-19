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
    DEFAULTS = {
        'sig2n_k': {'other': False, 'ols': True, 'gmlag': False, '2sls': True},
        'gmm': {'epsilon': 1e-05, 'inferenceOnLambda': True, 'max_iter': 1, 'step1c': False, 'inv_method': 'power_exp'},
        'instruments': {'lag_q': True, 'w_lags': 1},
        'other': {'ols_diagnostics': True, 'numcores': 1, 'residualMoran': False},
        'output': {'save_pred_residuals': False, 'vm_summary': False}}

    def __init__(self,parent=None):
        remapEvtsToDispatcher(self, self.evtDispatch)
        preferences_xrc.xrcgsPrefsFrame.__init__(self,parent)

        self.dispatch = d = {}
        d['saveButton'] = self.save
        self.SetPrefs(self.DEFAULTS)

    def __set_modified(self,val):
        self.__mod = bool(val)
        if self.__mod:
            self.SetTitle("GeoDaSpace Preferences*")
        else:
            self.SetTitle("GeoDaSpace Preferences")
    def __get_modified(self):
        return self.__mod
    modified = property(fget = __get_modified, fset = __set_modified)

    def evtDispatch(self,evtName, evt):
        evtName,widgetName = evtName.rsplit('_',1)
        if evtName not in ['restoreButton','saveButton','cancelButton']:
            self.modified = True
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
        print evtName, evt, value
        #print "OLS  stddev.:", self.OLSNk.GetValue(), self.OLSN.GetValue()
        #print "2SLS stddev.:", self.twoSLSNk.GetValue(), self.twoSLSN.GetValue()
        #print "GMlg stddev.:", self.GMlagNk.GetValue(), self.GMlagN.GetValue()
        #print "Othr stddev.:", False, self.othermodelsN.GetValue()
        print self.GetPrefs()
        print "hide dialog"
        return wx.ID_OK
    def SetPrefs(self,prefs):
        if 'sig2n_k' in prefs:
            sig = prefs['sig2n_k']
            # No option for other models.
            self.othermodelsN.SetValue(True)
            if 'ols' in sig:
                self.OLSNk.SetValue(bool(sig['ols']))
                self.OLSN.SetValue(not bool(sig['ols']))
            if '2sls' in sig:
                print sig['2sls']
                self.twoSLSNk.SetValue(bool(sig['2sls']))
                #self.twoSLSN.SetValue(not bool(sig['2sls']))
            if 'gmlag' in sig:
                self.twoSLSNk.SetValue(bool(sig['gmlag']))
                self.twoSLSN.SetValue(not bool(sig['gmlag']))
        
    def GetPrefs(self):
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
        s['output'] = {'vm_summary':self.ShowVarCovarMatrix.GetValue(), 'save_pred_residuals':self.saveValuesResiduals.GetValue()}
        s['other'] = {'ols_diagnostics':self.OLSdiagnostics.GetValue(), 'residualMoran':self.residualMoran.GetValue()}
        s['other']['numcores'] = self.numcores.GetValue()
        self._prefs = s
        return s
            

