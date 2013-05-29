import wx
from controls import C_CreateSpatialLag


class ddtApp(wx.App):
    def __init__(self, redirect=False, filename=None):
        wx.App.__init__(self, redirect, filename)

    def OnInit(self):
        """dataFile='path/to/file.txt', wtFiles=['f1','f2','f3'],
        vars=['var1','var2','var3','var4'])"""
        self.frame = C_CreateSpatialLag()
        self.frame.Show()
        return True
