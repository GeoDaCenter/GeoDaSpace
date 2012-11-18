import wx
from geodaspace.spatialLag.rc import SpatialLag_xrc
from geodaspace.abstractmodel import AbstractModel
from controls import C_CreateSpatialLag

class ddtApp(wx.App):
    def __init__(self,redirect=False,filename=None):
        wx.App.__init__(self,redirect,filename)
    def OnInit(self):
        self.frame = C_CreateSpatialLag()#dataFile='path/to/file.txt',wtFiles=['f1','f2','f3'],vars=['var1','var2','var3','var4'])
        #self.frame.addRow(varIDX=2)
        #self.frame.addRow(varIDX=0)
        #self.frame.addRow(varIDX=1)
        self.frame.Show()
        return True



if __name__=='__main__':
    dataFile = '/Users/charlie/Documents/repos/inclusio/trunk/inclusio/Data/Shapes/usa.dbf'
    app = ddtApp()
    app.MainLoop()

