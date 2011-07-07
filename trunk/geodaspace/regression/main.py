import wx
from geodaspace.regression import variableTools
from geodaspace.regression import M_regression
from geodaspace.regression import V_regression

"""
TODO:
    If Model Type == Standard and Endo == Yes, no OLS, yes IV
    IF spatial error, gmm, no HAC
"""

class guiApp(wx.App):
    """ Entry Point for GMM Regression GUI """
    results = []
    def __init__(self,redirect=False,filename=None):
        wx.App.__init__(self,redirect,filename)
    def OnInit(self):
        # Load the Frame
        self.regFrame = V_regression.guiRegView(results=self.results)
        self.regFrame.Show()
        self.SetTopWindow(self.regFrame)
        return True

if __name__=="__main__":
    """ usage: python -i main.py """
    app = guiApp()
    app.MainLoop()
