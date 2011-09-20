import wx
from control import preferencesDialog

class SimpleStandaloneApp(wx.App):
    def OnInit(self):
        self.dlg = preferencesDialog()
        #print "prefs:"
        #print self.dlg.GetPrefs()
        #print "show()"
        self.SetTopWindow(self.dlg)
        self.dlg.ShowModal()
        return False

def run():
    app = SimpleStandaloneApp(redirect=False)
    app.MainLoop()

if __name__ == '__main__':
    run()
