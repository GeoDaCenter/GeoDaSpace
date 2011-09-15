#system
#3rd Part
import wx
#local
from geodaspace import remapEvtsToDispatcher#,DEBUG
import preferences_xrc

DEBUG = True
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
        remapEvtsToDispatcher(self, self.evtDistpatch)
        preferences_xrc.xrcgsPrefsFrame.__init__(self,parent)

        self.dispatch = d = {}

    def evtDispatch(self,evtName, evt):
        evtName,widgetName = evtName.rsplit('_',1)
        if widgetName in self.dispatch:
            self.dispatch[widgetName](evtName,evt)
        else:
            if DEBUG: print "not implemented:", evtName, widgetName

