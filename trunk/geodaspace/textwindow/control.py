#Standard
import os.path
import StringIO
#Custom
import wx
#Local
import view

class TextWindow(view.xrcTextWindow,StringIO.StringIO):
    """ A Generic Text Container with, New, Open, Save and SaveAs buttons 
        model: TextWindow is it's own model, since it subclasses StringIO
    """
    def __init__(self,parent=None):
        view.xrcTextWindow.__init__(self,parent=parent)
        StringIO.StringIO.__init__(self)
        self.bindings()
        print "setting modified == false"
        self.Text.SetModified(False)
        #self.Text.SetModified(True)
        self.path = None
        self.populate()
    def bindings(self):
        self.Bind(wx.EVT_MENU, self.new, id = wx.xrc.XRCID("ToolNew"))
        self.Bind(wx.EVT_MENU, self.open, id = wx.xrc.XRCID("ToolOpen"))
        self.Bind(wx.EVT_MENU, self.save, id = wx.xrc.XRCID("ToolSave"))
        self.Bind(wx.EVT_MENU, self.saveAs, id = wx.xrc.XRCID("ToolSaveAs"))
        self.Bind(wx.EVT_MENU, self.bigger, id = wx.xrc.XRCID("ToolFontBigger"))
        self.Bind(wx.EVT_MENU, self.smaller, id = wx.xrc.XRCID("ToolFontSmaller"))
    def bigger(self,evt):
        font = self.Text.GetFont()
        font.SetPointSize(font.GetPointSize()+1)
        self.Text.SetFont(font)
    def smaller(self,evt):
        font = self.Text.GetFont()
        font.SetPointSize(font.GetPointSize()-1)
        self.Text.SetFont(font)
    def close(self):
        pass
    def title(self,default="Untitled"):
        if self.path:
            fName = os.path.split(self.path)[1]
            self.SetTitle(fName[:-4])
        else:
            self.SetTitle(default)
        if self.Text.IsModified():
            self.SetTitle(self.GetTitle()+'*')
        else:
            self.SetTitle(self.GetTitle().replace('*',''))
    def confirm(self):
        """ Prevents results from being lost be promting the user to save any changes that have been made,
            returns True, ok to continue
            or False, NOT ok to continue, if false do not distrube any results.
        """
        if self.Text.IsModified() or "*" in self.GetTitle(): #* in title is a bug fix for windows, some how Modified gets set to false!
            if not self.IsShown():
                self.Show()
            self.Raise()
            confirmDialog = wx.MessageDialog(self,"Your changes will be lost if you don't save them.",'Do you want to save the changes you made in "%s"?'%self.GetTitle().replace('*',''),style=wx.YES_NO|wx.CANCEL|wx.CENTRE)
            result = confirmDialog.ShowModal()
            if result == wx.ID_YES:
                self.save()
                return True
            elif result == wx.ID_NO:
                return True
            else: #wx.ID_CANCEL
                return False
        else:
            return True
    def new(self,evt=None):
        if self.confirm():
            self.seek(0)
            self.truncate()
            self.Text.SetModified(False)
            self.path = None
            self.populate()
    def open(self,evt=None):
        if self.confirm():
            #clear
            self.seek(0)
            self.truncate()
            fileDialog = wx.FileDialog(self,message="Select A File To Open...",wildcard="*.txt")
            result = fileDialog.ShowModal()
            if result == wx.ID_OK:
                path = fileDialog.GetPath()
                self.path = path
                f = open(path,'r')
                self.write(f.read())
                f.close()
                self.Text.SetModified(False)
                self.populate()
    def save(self,evt=None):
        if self.path:
            f = open(self.path,'w')
            self.seek(0)
            f.write(self.read())
            f.close()
            self.Text.SetModified(False)
            self.populate()
        else:
            self.saveAs()
    def saveAs(self,evt=None):
        fileDialog = wx.FileDialog(self,defaultFile=self.GetTitle().replace('*','')+'.txt',message="Save File As...",wildcard="*.txt",style=wx.SAVE+wx.OVERWRITE_PROMPT)
        result = fileDialog.ShowModal()
        if result == wx.ID_OK:
            path = fileDialog.GetPath()
            self.path= path
            self.save()
            self.populate()
    def write(self,s):
        StringIO.StringIO.write(self,s)
        self.Text.SetModified(True)
        self.populate()
    def populate(self):
        self.title()
        pos = self.Text.GetLastPosition()
        self.seek(0)
        self.Text.SetValue(self.read())
        self.Text.SetInsertionPoint(pos)
    def Close(self,evt=None):
        print "here"
        if self.confirm():
            print "confirm True!"
            return view.xrcTextWindow.Close(self)
            #self.Destroy()
        else:
            print "confirm False!"
            return False
