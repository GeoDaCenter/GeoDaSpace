import wx
import random
import sys
from geodaspace.regression.rc import OGRegression_xrc
from geodaspace.abstractmodel import AbstractModel


class mVariableSelector(AbstractModel):
    def __init__(self,data):
        AbstractModel.__init__(self)
    def getColumnNames(self):
        return ("Variable Names",)
    def getValues(self):
        return data['colNames']

class VarSearchCtrl(wx.SearchCtrl):
    def __init__(self,parent=None,id=-1,value="",pos=wx.DefaultPosition,size=wx.DefaultSize,style=0,doSearch=None):
        style |= wx.TE_PROCESS_ENTER
        wx.SearchCtrl.__init__(self, parent, id, value, pos, size, style)
        self.SetDescriptiveText('Filter')
        self.ShowCancelButton(True)
        self.inclusive = True
        #Bindings
        self.Bind(wx.EVT_TEXT_ENTER, self.OnTextEntered)
        self.Bind(wx.EVT_TEXT, self.OnTextEntered)
        self.Bind(wx.EVT_MENU_RANGE, self.OnMenuItem, id=1, id2=2)
        self.Bind(wx.EVT_SEARCHCTRL_CANCEL_BTN, self.clear)
        self.doSearch = doSearch
        self.SetMenu(self.MakeMenu())
    def OnTextEntered(self,evt):
        text = self.GetValue()
        if self.doSearch:
            self.doSearch(text,self.inclusive)
    def OnMenuItem(self,evt):
        if evt.GetId() == 1:
            self.inclusive = True
        else:
            self.inclusive = False
        self.OnTextEntered(None)
    def clear(self,evt=None):
        self.SetValue('')
    def MakeMenu(self):
        menu = wx.Menu()
        menu.Append(1, "Include Variables Containing...")
        menu.Append(2, "Exclude Variables Containing...")
        return menu

class vVariableSelector(wx.MiniFrame):
    #""" modified from http://wiki.wxpython.org/ListControls """
    def __init__(self,parent=None,style=wx.DEFAULT_FRAME_STYLE,size=(200,400),values=None):
        wx.MiniFrame.__init__(self,parent=parent,style=style,size=size)
        self.panel = OGRegression_xrc.xrcVariablePanel(self)
        self.sourcelist = self.panel.variableList
        self.sourcelist.Bind(wx.EVT_LIST_BEGIN_DRAG, self._startDrag)
        #self.panel.ToolBar.Bind(wx.EVT_MENU, self.spatialLag, id = wx.xrc.XRCID("ToolSpatialLag"))
        self.panel.ToolBar.EnableTool(wx.xrc.XRCID("ToolSpatialLag"),False)
        search = VarSearchCtrl(self.panel.ToolBar,size=(120,-1),doSearch=self.Search)
        #self.panel.ToolBar.InsertControl(self.panel.ToolBar.GetToolsCount(),search)
        self.panel.ToolBar.AddControl(search)
        self.panel.ToolBar.Realize()

        self.values = values
        self.populate()
        #self.panel.ToolBar.Hide()

    def Search(self,text,inclusive):
        if inclusive:
            self.populate([v for v in self.values if text.lower() in v.lower()])
        else:
            if text:
                self.populate([v for v in self.values if text.lower() not in v.lower()])
            else:
                self.populate(None)
    def populate(self,values=None):
        """ colNames = ('Col One','Col Two')
            values = [ (5,2) , (2,3) , ('string','4) ]
        """
        #for col in colNames:
        self.sourcelist.ClearAll()
        self.sourcelist.InsertColumn(1,"abcdefghijklmnopqrstuvwxyz")
        if values is not None:
            for item in values:
                self.sourcelist.InsertStringItem(sys.maxint, item)
        elif self.values:
            for item in self.values:
                self.sourcelist.InsertStringItem(sys.maxint, item)
        #else:
        #    for item in "abcdefghijklmnopqrstuvwxyz"*100:
        #        self.sourcelist.InsertStringItem(sys.maxint, item+'_%d'%(random.randint(80,99)))
        self.sourcelist.SetColumnWidth(0,wx.LIST_AUTOSIZE_USEHEADER)
        #dt = ListDrop(self.leftclick)

    def _startDrag(self,e):
        data = wx.PyTextDataObject()
        items = self.getSelected()
        items = ','.join(items)
        data.SetText(items)

        dropSource = wx.DropSource(self.sourcelist)
        dropSource.SetData(data)
        res = dropSource.DoDragDrop(flags=wx.Drag_CopyOnly)

    def getSelected(self):
        idx = -1
        items = []
        while True:
            idx = self.sourcelist.GetNextItem(idx,state=wx.LIST_STATE_SELECTED)
            if idx == -1:
                break
            else:
                items.append(self.sourcelist.GetItemText(idx))
        return items
            

class ddtApp(wx.App):
    def __init__(self,redirect=False,filename=None):
        wx.App.__init__(self,redirect,filename)
    def OnInit(self):
        values = [item+'_%d'%(random.randint(80,99)) for item in "abcdefghijklmnopqrstuvwxyz"*2]
        self.frame = vVariableSelector(values=values)
        self.frame.Show()
        return True



if __name__=='__main__':
    dataFile = '/Users/charlie/Documents/repos/inclusio/trunk/inclusio/Data/Shapes/usa.dbf'
    app = ddtApp()
    app.MainLoop()

