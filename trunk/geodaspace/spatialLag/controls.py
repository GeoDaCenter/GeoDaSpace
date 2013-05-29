import os.path
import wx
from geodaspace.spatialLag.rc import SpatialLag_xrc
#from geodaspace.abstractmodel import AbstractModel
from models import M_CreateSpatialLag, M_spLagVariable


class C_spLagVariable(SpatialLag_xrc.xrcSpLagVariable):
    """ Control for an XRC panel that contains, [textCtrl] = W*[dropDown] """
    def __init__(self, parent=None):
        SpatialLag_xrc.xrcSpLagVariable.__init__(self, parent)
        self.model = M_spLagVariable()
        self.model.addListener(self.populate)
        d = {}
        d['var'] = self.__var
        d['vars'] = self.__var
        d['newVarName'] = self.__newVarName
        d['caution'] = self.__caution
        self.dispatch = d
        self.varsChoice.Bind(
            wx.EVT_CHOICE, self.__var)  # AppendItems(self.model.get('vars'))
        self.newVarNameCtrl.Bind(
            wx.EVT_TEXT, self.__newVarName)  # SetValue(newVarName)
        # self.warn.Bind(wx.EVT_MOTION,self.info)
    # def info(self,evt):
    #    print evt
    #    print self.warn
    #    print dir(self.warn)

    def populate(self, model):
        data = self.model.get()
        for key in self.model.DATA_KEYS_ORDER:
            v = data[key]
            if key in self.dispatch:
                self.dispatch[key](value=v)
            else:
                print "Warning: %s, has not been implemented" % key
        self.able()

    def able(self):
        if self.model.get('var') == -1:
            self.newVarNameCtrl.Disable()
        else:
            self.newVarNameCtrl.Enable()

    def __var(self, evt=None, value=None):
        # print "__var...EVT: ",evt,"Value:",value
        if evt is not None:  # evt fired
            old_value = self.model.get('var')
            value = self.varsChoice.GetSelection()
            self.model.set('var', value)
            if not old_value == value:
                self.__newVarName(value=value)
        elif value is not None:  # model changed
            if type(value) == int:
                old_value = self.varsChoice.GetSelection()
                self.varsChoice.SetSelection(value)
                if value != -1 and old_value != value:
                    self.__newVarName(value=value)
                    # self.model.set('newVarName', self.model.get('vars')
                    #               [self.model.get('var')])
            else:  # assume list
                if self.model.varsChanged:
                    self.varsChoice.Clear()
                    if '' in value:
                        value.remove('')
                    self.varsChoice.AppendItems(value)
                    self.model.varsChanged = False

    def __newVarName(self, evt=None, value=None):
        # print "__newVarName...EVT: ",evt,"Value:",value
        if evt is not None:
            # pass
            if self.newVarNameCtrl.GetValue() != self.model.get('newVarName'):
                self.model.set('newVarName', self.newVarNameCtrl.GetValue())
        elif value is not None:
            if type(value) == int:  # change from __var
                # print self.model.get('var')
                # print self.model.get('vars')
                varName = self.model.get('vars')[self.model.get('var')]
                newVarName = 'W_' + varName
                self.model.set('newVarName', newVarName, passive=False)
            else:  # assume string
                self.warn.Show()
                # self.good.Show()
                self.newVarNameCtrl.SetValue(value)

    def __caution(self, evt=None, value=None):
        if value is not None:
            if self.model.get('newVarName') in self.model.get('vars'):
                self.warn.SetToolTipString(
                    "This variable already exists, and will be overwritten!")
                self.warn.Show()
                self.model.set('caution', True, True)
            elif len(self.model.get('newVarName')) < 1 and \
                    self.newVarNameCtrl.IsEnabled():
                self.warn.SetToolTipString(
                    "Please enter a variable name, or leave empty to exclude\
                    this variable.")
                self.warn.Show()
                self.model.set('caution', True, True)
            elif self.model.newVars.count(self.model.get('newVarName')) > 1:
                self.warn.SetToolTipString(
                    "You are creating another new variable with this name,\
                    please change one of them!")
                self.warn.Show()
                self.model.set('caution', True, True)
            elif len(self.model.get('newVarName')) > 10:
                self.warn.SetToolTipString(
                    "This variable name is too long. Truncated to 10 chars.")
                self.warn.Show()
                self.model.set('caution', True, True)
            else:
                self.warn.Hide()
                self.model.set('caution', False, True)


class C_CreateSpatialLag(SpatialLag_xrc.xrcCreateSpatialLag):
    def __init__(self, parent=None, dataFile=None, wtFiles=[''], vars=[''],
                 results=[], dialogMode=False):
        self.results = results
        self.dialogMode = dialogMode
        SpatialLag_xrc.xrcCreateSpatialLag.__init__(self, parent)
        self.width = None
        d = {}
        d['wtFiles'] = self.__wtFile
        d['wtFile'] = self.__wtFile
        d['vars'] = self.__vars
        d['dataFile'] = self.__dataFile
        self.dispatch = d

        self.weights.Bind(wx.EVT_CHOICE, self.__wtFile)
        self.fakeVarsChoice.Bind(wx.EVT_CHOICE, self.addRow)
        self.openWeights.Bind(wx.EVT_BUTTON, self.__wtFile)
        self.cancelButton.Bind(wx.EVT_BUTTON, self.close)
        self.okButton.Bind(wx.EVT_BUTTON, self.run)
        self.dataFileButton.Bind(wx.EVT_BUTTON, self.__dataFile)
        self.dataFile.Bind(wx.EVT_TEXT, self.__dataFile)

        self.model = M_CreateSpatialLag()
        self.model.addListener(self.populate)
        for wtFile in wtFiles:
            self.model.set('wtFile', wtFile)
        self.model.set('wtFile', 0)
        self.model.set('vars', vars)
        # hack to force initial data file to show up in GUI, even if it doesn't
        # exist
        self.model.data['dataFile'] = dataFile
        self.populate(None)
        self.model.set('dataFile', dataFile)

        # VariablesPeer is a hidden panel of size (0,0), this is necessary
        # because XRC does
        # not allow accessing sizers directly, so you get call a sizer item's
        # GetContainingSizer method
        self.varSizer = self.VariablesPeer.GetContainingSizer()
        # self.addRow()
        # print self.GetEffectiveMinSize()
        # self.SetMinSize(self.GetSize())

    def close(self, evt=None):
        self.Destroy()

    def run(self, evt):
        dpath, dfile = os.path.split(self.model.get('dataFile'))
        wild = '*.' + dfile.split('.')[1]
        fileDialog = wx.FileDialog(self, defaultFile=dfile, defaultDir=dpath,
                                   message="Save Data File As...",
                                   wildcard=wild, style=wx.SAVE +
                                   wx.OVERWRITE_PROMPT)
        result = fileDialog.ShowModal()
        if result == wx.ID_OK:
            path = fileDialog.GetPath()
            self.model.run(path)
            self.results.append(path)
            dialog = wx.MessageDialog(self,
                                      "The new variables were saved in, \n%s"
                                      % (path), "Success",
                                      wx.OK | wx.ICON_INFORMATION)
            dialog.ShowModal()
            self.model.set('dataFile', path)
            if self.dialogMode:
                self.close()

    def addRow(self, evt=None, varIDX=-1):
        """Add an additional row to the Variables Table"""
        if evt and evt.EventType == wx.EVT_CHOICE.typeId:
            varIDX = self.fakeVarsChoice.GetSelection()
            self.fakeVarsChoice.SetSelection(-1)

        # get old width,weight
        width, height = self.GetSize()
        var = C_spLagVariable(self.panel)
        if varIDX != -1:
            var.model.set('var', varIDX, passive=True)
        self.model.addVar(var.model)
        self.varSizer.Add(var, flag=wx.LEFT | wx.RIGHT | wx.EXPAND)
        self.varSizer.Layout()
        self.Fit()
        if not self.width:
            self.width, height = self.GetSize()

        self.SetMinSize((self.width, self.GetSize()[1]))
        self.SetSize((width, height))
        self.model.update()
        var.model.update()

    def populate(self, model):
        data = self.model.get()
        for key in self.model.DATA_KEYS_ORDER:
            v = data[key]
            if key in self.dispatch:
                self.dispatch[key](value=v)
            else:
                print "Warning: %s, has not been implemented" % key

    def __wtFile(self, evt=None, value=None):
        if evt:
            if evt.EventType == wx.EVT_CHOICE.typeId:  # The drop down
                self.model.set('wtFile', self.weights.GetSelection())
            elif evt.EventType == wx.EVT_BUTTON.typeId:
                print "button"
                filter = "Weights File (*.gal; *.gwt)|*.gal;*.gwt"
                # "|*.gal|GWT file|*.gwt|XML Weights|*.xml"
                fileDialog = wx.FileDialog(
                    self, message="Choose Weights File", wildcard=filter)
                result = fileDialog.ShowModal()
                if result == wx.ID_OK:
                    path = fileDialog.GetPath()
                    self.model.set('wtFile', path)
                else:
                    print "canceled"
        elif value is not None:
            if type(value) == int:
                self.weights.SetSelection(value)
            else:
                self.weights.Clear()
                if '' in value:
                    value.remove('')
                values = []
                for p in value:
                    if issubclass(type(p), basestring):
                        values.append(os.path.basename(p))
                    else:
                        values.append(p.name)
                self.weights.AppendItems(values)

    def __vars(self, evt=None, value=None):
        if value is not None:
            self.fakeVarsChoice.Clear()
            if '' in value:
                value.remove('')
            self.fakeVarsChoice.AppendItems(value)

    def __dataFile(self, evt=None, value=None):
        if evt:
            if evt.EventType == wx.EVT_BUTTON.typeId:
                filter = "Data File (*.dbf; *.csv)|*.dbf;*.csv"
                # "|*.gal|GWT file|*.gwt|XML Weights|*.xml"
                fileDialog = wx.FileDialog(
                    self, message="Choose Data File", wildcard=filter)
                result = fileDialog.ShowModal()
                if result == wx.ID_OK:
                    path = fileDialog.GetPath()
                    self.model.set('dataFile', path)
                else:
                    print "canceled"
            elif evt.EventType == wx.EVT_TEXT.typeId:
                if not value == self.dataFile.GetValue():
                    self.model.set(
                        'dataFile', self.dataFile.GetValue())  # ,passive=True)
        elif value is not None:
            if value is False:
                print "warning datafile"
                self.dataFileWarn.Show()
            else:
                if not value == self.dataFile.GetValue():
                    self.dataFile.SetValue(value)
                self.dataFileWarn.Hide()
