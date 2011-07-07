import wx
import control

class guiApp(wx.App):
    """ Entry Point for GMM Regression GUI """
    results = []
    def __init__(self,redirect=False,filename=None):
        wx.App.__init__(self,redirect,filename)
    def OnInit(self):
        # Load the Frame
        self.Frame = control.TextWindow()
        self.Frame.Show()
        self.SetTopWindow(self.Frame)
        return True

if __name__=="__main__":
    """ usage: python -i main.py """
    app = guiApp()
    app.MainLoop()
