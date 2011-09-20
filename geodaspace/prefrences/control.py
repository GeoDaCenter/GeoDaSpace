#system
import os
import json
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

INV_METHODS = ('power_exp','true_inverse',)
class preferencesDialog(preferences_xrc.xrcgsPrefsDialog):
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
        'sig2n_k': {'other': False, 'ols': True, 'gmlag': False, '2sls': False},
        'gmm': {'epsilon': 1e-05, 'inferenceOnLambda': True, 'max_iter': 1, 'step1c': False, 'inv_method': 'power_exp'},
        'instruments': {'lag_q': True, 'w_lags': 1},
        'other': {'ols_diagnostics': True, 'numcores': 1, 'residualMoran': False},
        'output': {'save_pred_residuals': False, 'vm_summary': False}}

    def __init__(self,parent=None):
        self.__mod = False
        remapEvtsToDispatcher(self, self.evtDispatch)
        preferences_xrc.xrcgsPrefsDialog.__init__(self,parent)
        self.CompInverse.SetItems(list(INV_METHODS))
        self.SetEscapeId(self.cancelButton.GetId())
        self.SetAffirmativeId(self.saveButton.GetId())

        self.dispatch = d = {}
        d['saveButton'] = self.save
        d['restoreButton'] = self.restore
        d['cancelButton'] = self.cancel
        if os.path.exists(self.config_file):
            config_fp = open(self.config_file,'r')
            prefs = json.load(config_fp)
            config_fp.close()
            try:
                self.SetPrefs(prefs)
            except:
                self.SetPrefs(self.DEFAULTS)
        else:
            self.SetPrefs(self.DEFAULTS)
        self.modified = False
    def OnClose(self,evt):
        return self.cancel(evt=evt)
    @property
    def config_file(self):
        paths = wx.StandardPaths_Get()
        return os.path.join(paths.GetUserConfigDir(),'GeoDaSpace.config')

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
        if widgetName not in ['restoreButton','saveButton','cancelButton']:
            print widgetName
            self.modified = True
        if widgetName in self.dispatch:
            self.dispatch[widgetName](evtName,evt)
        else:
            if DEBUG: print "not implemented:", evtName, widgetName

    def error(self, msg, tagline="An Error has occurred"):
        """ Display an error message to the user """
        dlg = wx.MessageDialog(self, msg, tagline, style=wx.OK|wx.ICON_ERROR).ShowModal()
        dlg.ShowModal()
        return
        
    def cancel(self, evtName=None, evt=None, value=None):
        print "hide dialog"
        if self.modified:
            dlg = wx.MessageDialog(self,"Unsaved changes will be lost.","Are you sure you wish to cancel?", style=wx.YES_NO|wx.ICON_QUESTION)
            if dlg.ShowModal() == wx.ID_YES:
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
            
    def save(self, evtName=None, evt=None, value=None):
        config = self.GetPrefs()
        config_fp = open(self.config_file,'w')
        json.dump(config,config_fp)
        config_fp.close()
        print config
        print "hide dialog"
        self.modified = False
        if self.IsModal():
            self.EndModal(wx.ID_OK)
        else:
            return wx.ID_OK
    def restore(self, evtName=None, evt=None, value=None):
        if self.modified:
            dlg = wx.MessageDialog(self,"All unsaved preferences will be lost.","Are you sure you wish to restore defaults?", style=wx.CANCEL|wx.OK|wx.ICON_QUESTION)
            if dlg.ShowModal() == wx.ID_OK:
                print 'ok to clear'
                self.SetPrefs(self.DEFAULTS)
        else:
            self.SetPrefs(self.DEFAULTS)
    def SetPrefs(self,prefs):
        """
        'sig2n_k': {'other': False, 'ols': True, 'gmlag': False, '2sls': False},
        'gmm': {'epsilon': 1e-05, 'inferenceOnLambda': True, 'max_iter': 1, 'step1c': False, 'inv_method': 'power_exp'},
        'instruments': {'lag_q': True, 'w_lags': 1},
        'other': {'ols_diagnostics': True, 'numcores': 1, 'residualMoran': False},
        'output': {'save_pred_residuals': False, 'vm_summary': False}}
        """
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
                self.twoSLSN.SetValue(not bool(sig['2sls']))
            if 'gmlag' in sig:
                self.GMlagNk.SetValue(bool(sig['gmlag']))
                self.GMlagN.SetValue(not bool(sig['gmlag']))
        if 'gmm' in prefs:
            gmm = prefs['gmm']
            if 'max_iter' in gmm:
                self.MaxIterations.SetValue(int(gmm['max_iter']))
            if 'epsilon' in gmm:
                self.StoppingCriterion.SetValue(str(float(gmm['epsilon'])))
            if 'inferenceOnLambda' in gmm:
                self.inferenceOnLambda.SetValue(bool(gmm['inferenceOnLambda']))
            if 'inv_method' in gmm:
                if gmm['inv_method'] in INV_METHODS:
                    self.CompInverse.SetSelection(INV_METHODS.index(gmm['inv_method']))
        self._prefs=prefs
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
            

