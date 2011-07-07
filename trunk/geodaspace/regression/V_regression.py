#Standard
import cPickle
import os.path
import StringIO
#Custom
import wx
#Local
from geodaspace import textwindow
from geodaspace.icons import icons
from geodaspace.regression.rc import OGRegression_xrc
from geodaspace import weights
from geodaspace.weights.control import ENABLE_CONTIGUITY_WEIGHTS, ENABLE_DISTANCE_WEIGHTS, ENABLE_KERNEL_WEIGTHS
from geodaspace import spatialLag
import variableTools
import M_regression

### NOTE: Tooltips are also set in the XRC for windows platforms, these only work on MAC
#MODEL_TYPE_TOOL_TIPS = {
#    0:"Estimates Standard OLS Model (No Space)", #Standard
#    1:"Estimated using S2SLS -Kelejian, H. and Prucha, I., (1998), Journal of Real Estate, Finance and Economics", #Spatial Lag
#    2:"Estimated using GM estimator -Kelejian, H. and Prucha, I., (1999), International Economic Review", #Spatial Error
#    3:"Estimated using GM estimator -Kelejian, H. and Prucha, I., (1999), International Economic Review", #Spatial Lag+Error
#    4:"This is the Tool Tip for Regimes", #Regimes
#}
#ENDOGENOUS_TOOL_TIPS = {
#    0:"Choose this option if you have additional endogenous variables within your set of explanatory variables. Drag your additional endogenous variables to the box labeled YE. Instruments for these variables should be defined in the box labeled H.", #Yes
#    1:"No endogenous variables within the set of explanatory variables.", #No
#}
WHITE_TOOL_TIP = "White Correction for Standard Errors"
HAC_TOOL_TIP = "Spatial Heteroskedastic and Autocorrelation Consistent Estimator for the Standard Errors -Kelejian, H. and Prucha, I. (2007), Journal of Econometrics" #HAC
#HET_TOOL_TIP = "Consistent Estimator Under Heteroskedastic Error Terms. -Kelejian , H. and Prucha, I. (Forthcoming), Journal of Econometrics" #HET
HET_TOOL_TIP = "Specification and Estimation of Spatial Autoregressive Models with Autoregressive and Heteroskedastic Disturbances - Kelejian, H and Prucha, I. (2010), Journal of Econometrics" #HET

myEVT_LIST_BOX_UPDATE = wx.NewEventType()
EVT_LIST_BOX_UPDATE = wx.PyEventBinder(myEVT_LIST_BOX_UPDATE, 1)

class MyStringIO(StringIO.StringIO):
    def close(self):
        pass
class TextCtrlDropTarget(wx.TextDropTarget):
    def __init__(self,target):
        wx.TextDropTarget.__init__(self)
        self.target = target
        #self.target.SetEditable(True)
        self.target.SetEditable(False)

        self.text_obj = wx.TextDataObject()
        self.SetDataObject(self.text_obj)
    def OnData(self,x,y,default): #Called on drop
        self.GetData()
        text = self.text_obj.GetText()

        #self.target.SetEditable(True)
        self.target.Clear()
        self.target.SetValue(text.split(',')[0])
        #self.target.SetEditable(False)
        if ',' in text:
            print "This field can only accept one value.  We'll take the first."
            print text
class NullDropTarget(wx.TextDropTarget):
    def __init__(self,targetListBox):
        wx.TextDropTarget.__init__(self)
        self.targetListBox = targetListBox
        self.text_obj = wx.TextDataObject()
        self.SetDataObject(self.text_obj)
    def OnData(self,x,y,default):
        return False
    def OnEnter(self,x,y,default):
        return wx.DragNone
    def OnDragOver(self,x,y,default):
        return wx.DragNone
        
class ListBoxDropTarget(wx.TextDropTarget):
    def __init__(self,targetListBox):
        wx.TextDropTarget.__init__(self)
        self.targetListBox = targetListBox

        self.text_obj = wx.TextDataObject()
        self.SetDataObject(self.text_obj)
    def OnData(self,x,y,default): #Called on drop
        if self.targetListBox.IsEnabled():
            self.GetData()
            text = self.text_obj.GetText()
            end_of_list = self.targetListBox.GetCount()
            self.targetListBox.InsertItems(text.split(','),end_of_list)
            # Fire my custom event type
            evt = ListBoxUpdateEvent(myEVT_LIST_BOX_UPDATE, self.targetListBox.GetId())
            self.targetListBox.GetEventHandler().ProcessEvent(evt)
        else:
            return False
    def OnEnter(self,x,y,default):
        if self.targetListBox.IsEnabled():
            return wx.DragCopy
        else:
            return wx.DragNone
    def OnDragOver(self,x,y,default):
        if self.targetListBox.IsEnabled():
            return wx.DragCopy
        else:
            return wx.DragNone

class ListBoxUpdateEvent(wx.PyCommandEvent):
    def __init__(self, evtType, id):
        """ A custom event type """
        wx.PyCommandEvent.__init__(self,evtType,id)

#class myTextFrame(OGRegression_xrc.xrcTextWindow,StringIO.StringIO):
#    """ A Generic Text Container with, New, Open, Save and SaveAs buttons """
#    def __init__(self,parent=None):
#        OGRegression_xrc.xrcTextWindow.__init__(self,parent=parent)
#        StringIO.StringIO.__init__(self)
#        self.bindings()
#        print "setting mod false"
#        self.Text.SetModified(False)
#        #self.Text.SetModified(True)
#        self.path = None
#        self.populate()
#    def bindings(self):
#        self.ToolBar.Bind(wx.EVT_MENU, self.new, id = wx.xrc.XRCID("ToolNew"))
#        self.ToolBar.Bind(wx.EVT_MENU, self.open, id = wx.xrc.XRCID("ToolOpen"))
#        self.ToolBar.Bind(wx.EVT_MENU, self.save, id = wx.xrc.XRCID("ToolSave"))
#        self.ToolBar.Bind(wx.EVT_MENU, self.saveAs, id = wx.xrc.XRCID("ToolSaveAs"))
#        self.ToolBar.Bind(wx.EVT_MENU, self.bigger, id = wx.xrc.XRCID("ToolFontBigger"))
#        self.ToolBar.Bind(wx.EVT_MENU, self.smaller, id = wx.xrc.XRCID("ToolFontSmaller"))
#    def bigger(self,evt):
#        font = self.Text.GetFont()
#        font.SetPointSize(font.GetPointSize()+1)
#        self.Text.SetFont(font)
#    def smaller(self,evt):
#        font = self.Text.GetFont()
#        font.SetPointSize(font.GetPointSize()-1)
#        self.Text.SetFont(font)
#    def close(self):
#        pass
#    def title(self,default="Untitled"):
#        if self.path:
#            fName = os.path.split(self.path)[1]
#            self.SetTitle(fName[:-4])
#        else:
#            self.SetTitle(default)
#        if self.Text.IsModified():
#            self.SetTitle(self.GetTitle()+'*')
#        else:
#            self.SetTitle(self.GetTitle().replace('*',''))
#    def confirm(self):
#        """ Prevents results from being lost be promting the user to save any changes that have been made,
#            returns True, ok to continue
#            or False, NOT ok to continue, if false do not distrube any results.
#        """
#        if self.Text.IsModified() or "*" in self.GetTitle(): #* in title is a bug fix for windows, some how Modified gets set to false!
#            if not self.IsShown():
#                self.Show()
#            self.Raise()
#            confirmDialog = wx.MessageDialog(self,"Your changes will be lost if you don't save them.",'Do you want to save the changes you made in "%s"?'%self.GetTitle().replace('*',''),style=wx.YES_NO|wx.CANCEL|wx.CENTRE)
#            result = confirmDialog.ShowModal()
#            if result == wx.ID_YES:
#                self.save()
#                return True
#            elif result == wx.ID_NO:
#                return True
#            else: #wx.ID_CANCEL
#                return False
#        else:
#            return True
#    def new(self,evt=None):
#        if self.confirm():
#            self.seek(0)
#            self.truncate()
#            self.Text.SetModified(False)
#            self.path = None
#            self.populate()
#    def open(self,evt=None):
#        if self.confirm():
#            fileDialog = wx.FileDialog(self,message="Select A File To Open...",wildcard="*.txt")
#            result = fileDialog.ShowModal()
#            if result == wx.ID_OK:
#                path = fileDialog.GetPath()
#                self.path = path
#                f = open(path,'r')
#                self.write(f.read())
#                f.close()
#                self.Text.SetModified(False)
#                self.populate()
#    def save(self,evt=None):
#        if self.path:
#            f = open(self.path,'w')
#            self.seek(0)
#            f.write(self.read())
#            f.close()
#            self.Text.SetModified(False)
#            self.populate()
#        else:
#            self.saveAs()
#    def saveAs(self,evt=None):
#        fileDialog = wx.FileDialog(self,defaultFile=self.GetTitle().replace('*','')+'.txt',message="Save File As...",wildcard="*.txt",style=wx.SAVE+wx.OVERWRITE_PROMPT)
#        result = fileDialog.ShowModal()
#        if result == wx.ID_OK:
#            path = fileDialog.GetPath()
#            self.path= path
#            self.save()
#            self.populate()
#    def write(self,s):
#        StringIO.StringIO.write(self,s)
#        self.Text.SetModified(True)
#        self.populate()
#    def populate(self):
#        self.title()
#        pos = self.Text.GetLastPosition()
#        self.seek(0)
#        self.Text.SetValue(self.read())
#        self.Text.SetInsertionPoint(pos)
#    def Close(self,evt=None):
#        print "here"
#        if self.confirm():
#            print "confirm True!"
#            return OGRegression_xrc.xrcTextWindow.Close(self)
#            #self.Destroy()
#        else:
#            print "confirm False!"
#            return False
#
#
class guiRegView(OGRegression_xrc.xrcGMM_REGRESSION):
    def __init__(self,parent=None,results=None):
        self.results = results
        OGRegression_xrc.xrcGMM_REGRESSION.__init__(self,parent)
        self.SetIcon(icons.getGeoDaIcon())
        self.modelFileName = None
        self.BASE_TITLE = self.GetTitle()
        print "BASE_TITLE: ",self.BASE_TITLE
        self.CreateStatusBar()
        self.GetStatusBar().SetStatusText("Welcome to GeoDaSpace")

        #initialize the scrollbars, fix for issue #48
        w,h = self.Size
        self.scroll.SetScrollbars(1,1,w,h)

        ### NOTE: Tooltips are also set in the XRC for windows platforms, these only work on MAC
        self.SEHACCheckBox.SetToolTipString(HAC_TOOL_TIP)
        self.SEHETCheckBox.SetToolTipString(HET_TOOL_TIP)
        self.SEWhiteCheckBox.SetToolTipString(WHITE_TOOL_TIP)
        #for i,radioButton in enumerate(self.ModelTypeRadioBox.GetChildren()):
        #    if i in MODEL_TYPE_TOOL_TIPS:
        #        radioButton.SetToolTipString(MODEL_TYPE_TOOL_TIPS[i])
        #for i,radioButton in enumerate(self.EndogenousRadioBox.GetChildren()):
        #    if i in ENDOGENOUS_TOOL_TIPS:
        #        radioButton.SetToolTipString(ENDOGENOUS_TOOL_TIPS[i])

        # Setup the Weights Dialog for future use, remember to clear the queue after use.
        self.wQueue = []
        #self.weightsFrame = weights.control.mxWeightsControl(self,results=self.wQueue)
        self.modelWeightsDialog = weights.control.weightsDialog(self,requireSave = False, style = ENABLE_CONTIGUITY_WEIGHTS|ENABLE_DISTANCE_WEIGHTS)
        self.kernelWeightsDialog = weights.control.weightsDialog(self,requireSave = False, style = ENABLE_KERNEL_WEIGTHS)
        #self.resultText = MyStringIO()
        self.textFrame = textwindow.TextWindow(self)
        #self.textFrame = myTextFrame(self)
        #self.textFrame.Show()

        self.varSelector = None
        self.createSpatialLag = None
        # Setup Drop Targets
        self.Y_TextCtrl.SetDropTarget(TextCtrlDropTarget(self.Y_TextCtrl))
        self.YE_ListBox.SetDropTarget(ListBoxDropTarget(self.YE_ListBox))
        self.H_ListBox.SetDropTarget(ListBoxDropTarget(self.H_ListBox))

        #self.R_TextCtrl.SetDropTarget(TextCtrlDropTarget(self.R_TextCtrl))
        self.R_TextCtrl.SetDropTarget(NullDropTarget(self.R_TextCtrl))
        #self.S_TextCtrl.SetDropTarget(TextCtrlDropTarget(self.S_TextCtrl))
        self.S_TextCtrl.SetDropTarget(NullDropTarget(self.S_TextCtrl))
        #self.T_TextCtrl.SetDropTarget(TextCtrlDropTarget(self.T_TextCtrl))
        self.T_TextCtrl.SetDropTarget(NullDropTarget(self.T_TextCtrl))

        self.X_ListBox.SetDropTarget(ListBoxDropTarget(self.X_ListBox))
        # The Model
        self.model = M_regression.guiRegModel()
        self.model.addListener(self.populate)
        #self.model.addListener(self.verbose)
        # The Bindings
        self.Bind(wx.EVT_BUTTON,self.DataOpenButtonClick,self.DATA_INPUTFILE)
        self.Bind(wx.EVT_BUTTON,self.OpenMWeightsButtonClick,self.OpenMWeightsButton)
        self.Bind(wx.EVT_BUTTON,self.OpenKWeightsButtonClick,self.OpenKWeightsButton)
        self.Bind(wx.EVT_BUTTON,self.CreateMWeightsButtonClick,self.CreateMWeightsButton)
        self.Bind(wx.EVT_BUTTON,self.CreateKWeightsButtonClick,self.CreateKWeightsButton)
        self.Bind(wx.EVT_BUTTON,self.run,self.RunButton)
        self.Bind(wx.EVT_BUTTON,self.saveModel,self.SaveButton)
        self.Bind(wx.EVT_BUTTON,self.close,self.CloseButton)
        #self.DATAFILE.Bind(wx.EVT_COMBOBOX,self.setDataFile)
        #self.DATAFILE.Bind(wx.EVT_TEXT_ENTER,self.setDataFile)
        #self.IDVAR.Bind(wx.EVT_CHOICE,self.setIDVar)

        self.Y_TextCtrl.Bind(wx.EVT_TEXT, self.updateSpec)
        self.Y_TextCtrl.Bind(wx.EVT_LEFT_DCLICK,self.clearTextBox)
        self.YE_ListBox.Bind(EVT_LIST_BOX_UPDATE, self.updateSpec)
        self.YE_ListBox.Bind(wx.EVT_LISTBOX_DCLICK, self.removeSelected)
        self.H_ListBox.Bind(EVT_LIST_BOX_UPDATE, self.updateSpec)
        self.H_ListBox.Bind(wx.EVT_LISTBOX_DCLICK, self.removeSelected)
        #self.R_TextCtrl.Bind(wx.EVT_TEXT, self.updateSpec)
        #self.R_TextCtrl.Bind(wx.EVT_LEFT_DCLICK,self.clearTextBox)
        #self.S_TextCtrl.Bind(wx.EVT_TEXT, self.updateSpec)
        #self.S_TextCtrl.Bind(wx.EVT_LEFT_DCLICK,self.clearTextBox)
        #self.T_TextCtrl.Bind(wx.EVT_TEXT, self.updateSpec)
        #self.T_TextCtrl.Bind(wx.EVT_LEFT_DCLICK,self.clearTextBox)
        self.X_ListBox.Bind(EVT_LIST_BOX_UPDATE, self.updateSpec)
        self.X_ListBox.Bind(wx.EVT_LISTBOX_DCLICK, self.removeSelected)

        # The next 2 lines are commented out to disable removing weights files.
        # TODO: the removal mechanism doesn't work with w objs, need to fix 
        #self.MWeights_ListBox.Bind(wx.EVT_LISTBOX_DCLICK, self.removeSelected)
        #self.KWeights_ListBox.Bind(wx.EVT_LISTBOX_DCLICK, self.removeSelected)

        #self.Bind(wx.EVT_RADIOBUTTON,self.updateModelType)
        
        self.MT_STD.Bind(wx.EVT_RADIOBUTTON,self.updateModelType)
        self.MT_LAG.Bind(wx.EVT_RADIOBUTTON,self.updateModelType)
        self.MT_ERR.Bind(wx.EVT_RADIOBUTTON,self.updateModelType)
        self.MT_LAGERR.Bind(wx.EVT_RADIOBUTTON,self.updateModelType)
        #self.ModelTypeRadioBox.Bind(wx.EVT_RADIOBOX, self.updateModelType)
        self.ENDO_CHECK.Bind(wx.EVT_CHECKBOX, self.updateModelType)
        #self.EndogenousRadioBox.Bind(wx.EVT_RADIOBOX, self.updateModelType)
        #self.MethodsRadioBox.Bind(wx.EVT_RADIOBOX, self.updateModelType)
        #self.SEClassicCheckBox.Bind(wx.EVT_CHECKBOX, self.updateModelType)
        self.SEWhiteCheckBox.Bind(wx.EVT_CHECKBOX, self.updateModelType)
        self.SEHACCheckBox.Bind(wx.EVT_CHECKBOX, self.updateModelType)
        self.SEHETCheckBox.Bind(wx.EVT_CHECKBOX, self.updateModelType)
        self.ST_LM.Bind(wx.EVT_CHECKBOX, self.updateModelType)
        
        self.RegressionToolBar.Bind(wx.EVT_MENU, self.newModel, id = wx.xrc.XRCID("ToolNewModel"))
        self.RegressionToolBar.Bind(wx.EVT_MENU, self.openModel, id = wx.xrc.XRCID("ToolOpenModel"))
        self.RegressionToolBar.Bind(wx.EVT_MENU, self.saveModel, id = wx.xrc.XRCID("ToolSaveModel"))
        self.RegressionToolBar.Bind(wx.EVT_MENU, self.saveModelAs, id = wx.xrc.XRCID("ToolSaveModelAs"))
        self.RegressionToolBar.Bind(wx.EVT_MENU, self.showVarSelector, id = wx.xrc.XRCID("ToolVariableSelector"))
        self.RegressionToolBar.Bind(wx.EVT_MENU, self.openResultsWindow, id = wx.xrc.XRCID("ToolResultsWindow"))

        self.textFrame.Bind(wx.EVT_CLOSE,self.openResultsWindow)
        self.Bind(wx.EVT_CLOSE,self.close)

        self.MODELTYPES = [ self.MT_STD,
                            self.MT_LAG,
                            self.MT_ERR,
                            self.MT_LAGERR ]
        self.populate(None)

    def close(self,evt=None):
        print "close"
        if self.textFrame.Close():
            self.Destroy()
    def able(self):
        #self.tooltips()
        """ Disables and/or Enables parts of the form, based on the status of the model """
        if self.model.state == self.model.STATE_EMPTY:
            #self.Panel.Disable()
            self.Panel_Weights.Disable()
            self.Panel_Spec.Disable()
            self.Panel_Estimation.Disable()
            self.RegressionToolBar.EnableTool(wx.xrc.XRCID("ToolVariableSelector"),False)
            self.RegressionToolBar.EnableTool(wx.xrc.XRCID("ToolResultsWindow"),False)
            self.RegressionToolBar.EnableTool(wx.xrc.XRCID("ToolSaveModel"),False)
            self.RegressionToolBar.EnableTool(wx.xrc.XRCID("ToolSaveModelAs"),False)
        else:
            #self.Panel.Enable()
            self.Panel_Weights.Enable()
            self.Panel_Spec.Enable()
            self.Panel_Estimation.Enable()
            self.RegressionToolBar.EnableTool(wx.xrc.XRCID("ToolVariableSelector"),True)
            self.RegressionToolBar.EnableTool(wx.xrc.XRCID("ToolResultsWindow"),True)
            self.RegressionToolBar.EnableTool(wx.xrc.XRCID("ToolSaveModel"),True)
            self.RegressionToolBar.EnableTool(wx.xrc.XRCID("ToolSaveModelAs"),True)
            # Not implemented yet
            #self.ModelTypeRadioBox.EnableItem(4,False) #Regimes.Disable()
            self.R_TextCtrl.Disable()
            self.S_TextCtrl.Disable()
            self.T_TextCtrl.Disable()

            # Fixed, always on.
            #self.SEClassicCheckBox.Disable()
            #self.SEWhiteCheckBox.Disable()

            #self.EnableMethods()
            m = self.model.data
            if m['fname']:
                if m['modelType']['endogenous'] == True: #Yes
                    self.YE_ListBox.Enable()
                    self.H_ListBox.Enable()
                else: #No
                    self.YE_ListBox.Disable()
                    self.H_ListBox.Disable()

                if m['modelType']['mType'] == 2 or m['modelType']['mType'] == 3: # an error model
                    #No White in Error Models
                    self.SEWhiteCheckBox.Disable()
                    self.SEWhiteCheckBox.SetValue(False)
                    m['modelType']['error']['white'] = False
                    # allow HET only in Error models
                    self.SEHETCheckBox.Enable()
                    self.ST_LM.Disable()
                    self.ST_LM.SetValue(False)
                else:
                    self.ST_LM.Enable()
                    self.SEWhiteCheckBox.Enable()
                    self.SEHETCheckBox.Disable()
                    self.SEHETCheckBox.SetValue(False)
                    m['modelType']['error']['het'] = False

                if m['modelType']['mType'] == 0 or m['modelType']['mType'] == 1: #Standard or SpatialLag
                    self.SEHACCheckBox.Enable()
                else:
                    self.SEHACCheckBox.SetValue(False)
                    m['modelType']['error']['hac'] = False
                    self.SEHACCheckBox.Disable()
            # model.verify is now called on Run, and an error msg displays why the model won't run.
            #if self.model.verify():
            #    self.RunButton.Enable()
            #else:
            #    self.RunButton.Disable()

    def setTitle(self):
        if self.modelFileName:
            fname = os.path.split(self.modelFileName)[1]
            if self.model.state == self.model.STATE_CHANGED:
                fname+="*"
            self.SetTitle(self.BASE_TITLE+": %s"%fname)
        elif self.model.state == self.model.STATE_CHANGED:
            self.SetTitle(self.BASE_TITLE+": untitled*")
    def newModel(self,evt):
        self.model.reset()
        self.modelFileName = None
        self.setTitle()
        self.DataOpenButtonClick(evt)
        
    def saveModelAs(self,evt):
        fname = self.model.data['fname']
        suggestion = os.path.split(fname)[1].split('.')[0]+'.mdl'
        fileDialog = wx.FileDialog(self,defaultFile=suggestion,message="Save Model As...",wildcard="*.mdl",style=wx.SAVE+wx.OVERWRITE_PROMPT)
        result = fileDialog.ShowModal()
        if result == wx.ID_OK:
            path = fileDialog.GetPath()
            if not path[-4:] == '.mdl':
                path+='.mdl'
            f = open(path,'w')
            self.model.save(f)
            f.close()
            self.modelFileName = path
            self.setTitle()
            print "saveModelAs, path...",path
        else:
            print "canceled"
    def saveModel(self,evt):
        if self.modelFileName:
            f = open(self.modelFileName,'w')
            self.model.save(f)
            f.close()
            self.setTitle()
        else:
            self.saveModelAs(evt)
            
    def locateMissingFile(self,oldPath):
        confirmDialog = wx.MessageDialog(self,"%s does not exist, would you like to locate the correct file?"%oldPath,'The file could not be located!',style=wx.YES_NO|wx.CENTRE)
        result = confirmDialog.ShowModal()
        if result == wx.ID_YES:
            if '\\' in oldPath:
                basename = oldPath.split('\\')[-1]
            elif '/' in oldPath:
                basename = oldPath.split('/')[-1]
            else:
                basename = os.path.basename(oldPath)
            filter = "%s|%s"%(basename,basename)
            fileDialog = wx.FileDialog(self,message="Please locate %s"%basename,wildcard=filter)
            fdResult = fileDialog.ShowModal()
            if fdResult == wx.ID_OK:
                path = fileDialog.GetPath()
                return path
        else:
            return False
        
    def openModel(self,evt):
        try:
            fileDialog = wx.FileDialog(self,message="Select Model To Open...",wildcard="*.mdl")
            result = fileDialog.ShowModal()
            if result == wx.ID_OK:
                path = fileDialog.GetPath()
                f = open(path,'r')
                dat = f.read()
                changed = False
                dataDict = eval(dat)
                dataFilePath = dataDict['fname']
                if not os.path.exists(dataFilePath):
                    print "no such file!"
                    newPath = self.locateMissingFile(dataFilePath)
                    if newPath:
                        dataDict['fname'] = newPath
                        dat = str(dataDict)
                        changed = True
                    else:
                        return False
                for i,mw in enumerate(dataDict['mWeights']):
                    if not os.path.exists(mw):
                        print "no such file!"
                        newPath = self.locateMissingFile(mw)
                        if newPath:
                            dataDict['mWeights'][i] = newPath
                            dat = str(dataDict)
                            changed = True
                        else:
                            return False
                for i,kw in enumerate(dataDict['kWeights']):
                    if not os.path.exists(kw):
                        print "no such file!"
                        newPath = self.locateMissingFile(kw)
                        if newPath:
                            dataDict['kWeights'][i] = newPath
                            dat = str(dataDict)
                            changed = True
                        else:
                            return False
                    
                        
                self.model.open(dat)
                f.close()
                self.modelFileName = path
                self.setTitle()
                if changed:
                    self.model.update() #sets modified flag
                print "openModel, path... ",path
            else:
                print "canceled"
        except:
            confirmDialog = wx.MessageDialog(self,"The model file could not be loaded.",'Invalid Model File!',style=wx.OK|wx.CENTRE)
            result = confirmDialog.ShowModal()
            self.model.reset()
            self.populate(self.model)
            #raise

    def updateModelType(self,evt):
        modelSetup = {}
        modelSetup['mType'] = [m.GetValue() for m in self.MODELTYPES].index(True)
        #modelSetup['mType'] = self.ModelTypeRadioBox.GetSelection() # Zero Base
        #modelSetup['endogenous'] = self.EndogenousRadioBox.GetSelection()
        modelSetup['endogenous'] = self.ENDO_CHECK.GetValue()
        #modelSetup['method'] = self.MethodsRadioBox.GetSelection()
        modelSetup['error'] = {}
        modelSetup['error']['classic'] = True #self.SEClassicCheckBox.GetValue() #True or False
        modelSetup['error']['white'] = self.SEWhiteCheckBox.GetValue()
        modelSetup['error']['hac'] = self.SEHACCheckBox.GetValue()
        modelSetup['error']['het'] = self.SEHETCheckBox.GetValue()
        modelSetup['spatial_tests'] = {}
        modelSetup['spatial_tests']['lm'] = self.ST_LM.GetValue()
        self.model.setModelType(modelSetup)
    def setModelType(self,setup):
        """ Compares the ModelType settings in self.model with the settings in the GUI form...
            These should only differ when model is loaded from file or changed programatically """
        #if not setup['mType'] == self.ModelTypeRadioBox.GetSelection():
        if not setup['mType'] == [m.GetValue() for m in self.MODELTYPES].index(True):
            self.MODELTYPES[setup['mType']].SetValue(True)
            #self.ModelTypeRadioBox.SetSelection(setup['mType'])
        #if not setup['endogenous'] == self.EndogenousRadioBox.GetSelection():
        if not setup['endogenous'] == self.ENDO_CHECK.GetValue():
            #self.EndogenousRadioBox.SetSelection(setup['endogenous'])
            self.ENDO_CHECK.SetValue(setup['endogenous'])
        #if not setup['method'] == self.MethodsRadioBox.GetSelection():
        #    self.MethodsRadioBox.SetSelection(setup['method'])
        #if not setup['error']['classic'] == self.SEClassicCheckBox.GetValue():
        #    self.SEClassicCheckBox.SetValue(setup['error']['classic'])
        if not setup['error']['white'] == self.SEWhiteCheckBox.GetValue():
            self.SEWhiteCheckBox.SetValue(setup['error']['white'])
        if not setup['error']['hac'] == self.SEHACCheckBox.GetValue():
            self.SEHACCheckBox.SetValue(setup['error']['hac'])
        if not setup['error']['het'] == self.SEHETCheckBox.GetValue():
            self.SEHACCheckBox.SetValue(setup['error']['het'])
        if not setup['spatial_tests']['lm'] == self.ST_LM.GetValue():
            self.ST_LM.SetValue(setup['spatial_tests']['lm'])

    def updateSpec(self,evt):
        #print "updateSpec, evt... ",evt
        spec = {}
        spec['y'] = self.Y_TextCtrl.GetValue()
        spec['YE'] = self.YE_ListBox.GetItems()
        spec['H'] = self.H_ListBox.GetItems()
        #spec['R'] = self.R_TextCtrl.GetValue()
        #spec['S'] = self.S_TextCtrl.GetValue()
        #spec['T'] = self.T_TextCtrl.GetValue()
        spec['X'] = self.X_ListBox.GetItems()
        #print "Setting Model Spec as... ",spec
        #print "form X_ListBox contains, ",self.X_ListBox.GetItems()
        #print "form X_ListBox contains Strings, ",self.X_ListBox.GetStrings()

        mwFiles = self.MWeights_ListBox.GetItems()
        newWeights = []
        for p in mwFiles:
            tmp = [w for w in self.model.data['mWeights'] if p in w]
            if tmp:
                newWeights.append(tmp[0])
        self.model.data['mWeights'] = newWeights

        kwFiles = self.KWeights_ListBox.GetItems()
        newWeights = []
        for p in kwFiles:
            tmp = [w for w in self.model.data['kWeights'] if p in w]
            if tmp:
                newWeights.append(tmp[0])
        self.model.data['kWeights'] = newWeights
            
        self.model.setSpec(spec)
            
    def setSpec(self,spec):
        #print "Setting Form Spec as... ",spec
        if not spec['y'] == self.Y_TextCtrl.GetValue():
            # TextCtrl.SetValue triggers an event, which interupts the form update.
            self.Y_TextCtrl.SetEvtHandlerEnabled(False)
            self.Y_TextCtrl.SetValue(spec['y'])
            self.Y_TextCtrl.SetEvtHandlerEnabled(True)
        if not spec['YE'] == self.YE_ListBox.GetItems():
            self.YE_ListBox.SetItems(spec['YE'])
        if not spec['H'] == self.H_ListBox.GetItems():
            self.H_ListBox.SetItems(spec['H'])
        #if not spec['R'] == self.R_TextCtrl.GetValue():
        #    self.R_TextCtrl.SetValue(spec['R'])
        #if not spec['S'] == self.S_TextCtrl.GetValue():
        #    self.S_TextCtrl.SetValue(spec['S'])
        #if not spec['T'] == self.T_TextCtrl.GetValue():
        #    self.T_TextCtrl.SetValue(spec['T'])
        if not spec['X'] == self.X_ListBox.GetItems():
            self.X_ListBox.SetItems(spec['X'])

    def openResultsWindow(self,evt=None):
        if self.textFrame.IsShown():
            self.textFrame.Hide()
        else:
            self.textFrame.Show()
            self.textFrame.Raise()

    def newVarSelector(self,evt=None):
        """ Util function, recreates the Variable Selector as needed """
        if self.varSelector:
            self.varSelector.Hide()
            self.varSelector.Destroy()
        self.varSelector = variableTools.vVariableSelector(self,values=self.model.getVariables())
        self.varSelector.Bind(wx.EVT_CLOSE,self.showVarSelector)
        self.varSelector.panel.ToolBar.Bind(wx.EVT_MENU, self.spatialLag, id = wx.xrc.XRCID("ToolSpatialLag"))
        self.varSelector.panel.ToolBar.EnableTool(wx.xrc.XRCID("ToolSpatialLag"),True)
        #self.varSelector.populate(self.model.getVariables())
        self.varSelector.CenterOnParent()
        self.varSelector.Show()
    def spatialLag(self,evt):
        if self.createSpatialLag:
            self.createSpatialLag.Hide()
            self.createSpatialLag.Destroy()
        print "here"
        print self.varSelector.getSelected()
        weights = set(self.model.data['mWeights'])
        weights = weights.union(set(self.model.data['kWeights']))
        print weights
        vars = self.model.getVariables()
        self.spLagQueue = []
        self.createSpatialLag = spatialLag.C_CreateSpatialLag(dataFile=self.model.data['fname'],wtFiles=weights,vars=vars,results=self.spLagQueue,dialogMode=True)
        for var in self.varSelector.getSelected():
            self.createSpatialLag.addRow(varIDX=vars.index(var))
        
        if self.varSelector.IsShown():
            self.showVarSelector()
            self.createSpatialLag.ShowModal()
            self.showVarSelector()
        else:
            self.createSpatialLag.ShowModal()

        if len(self.spLagQueue) >= 1:
            if os.path.exists(self.spLagQueue[-1]):
                self.setDataFile(self.spLagQueue[-1],True)
    def showVarSelector(self,evt=None):
        """ Util function, hides/shows the Variable Selector as needed """
        if not self.varSelector:
            self.newVarSelector()
            self.varSelector.Hide()
        if self.varSelector.IsShown():
            self.varSelector.Hide()
        else:
            self.varSelector.Show()
            self.varSelector.Raise()
            

    def clearTextBox(self,evt):
        " Clears a textbox, bind to double click "
        target = evt.EventObject
        target.Clear()
        self.updateSpec(evt)
    def removeSelected(self,evt):
        " Removes a selected item from listbox, bind to double click "
        target = evt.EventObject
        target.Delete(evt.Selection)
        self.updateSpec(evt)

    def CreateMWeightsButtonClick(self,evt):
        f = self.model.data['fname']
        f = f[:-3]+'shp'
        if os.path.exists(f):
            try:
                #self.weightsFrame.model.set('inShp',f)
                self.modelWeightsDialog.model.inShp = f
            except:
                raise
        if self.modelWeightsDialog.ShowModal() == wx.ID_OK:
            self.model.addMWeightsFile(obj=self.modelWeightsDialog.GetW())
        #print self.wQueue
        #while self.wQueue:
        #    path = self.wQueue.pop(0)
        #    self.model.addMWeightsFile(path=path)
    def OpenMWeightsButtonClick(self,evt):
        pathHint = os.path.split(self.model.data['fname'])[0]
        filter = "Weights File (*.gal; *.gwt)|*.gal;*.gwt" #"|*.gal|GWT file|*.gwt|XML Weights|*.xml"
        fileDialog = wx.FileDialog(self,defaultDir=pathHint,message="Choose Weights File",wildcard=filter)
        result = fileDialog.ShowModal()
        if result == wx.ID_OK:
            path = fileDialog.GetPath()
            self.model.addMWeightsFile(path=path)
        else:
            print "canceled"
    def CreateKWeightsButtonClick(self,evt):
        f = self.model.data['fname']
        f = f[:-3]+'shp'
        if os.path.exists(f):
            try:
                self.weightsFrame.model.set('inShp',f)
            except:
                raise
        self.weightsFrame.ShowModal()
        print self.wQueue
        while self.wQueue:
            path = self.wQueue.pop(0)
            self.model.addKWeightsFile(path=path)
            self.KernelWeightsWarning(path)
    def OpenKWeightsButtonClick(self,evt):
        pathHint = os.path.split(self.model.data['fname'])[0]
        filter = "Weights File (*.gal; *.gwt)|*.gal;*.gwt" #"|*.gal|GWT file|*.gwt|XML Weights|*.xml"
        #filter = "GWT file|*.gwt|GAL file|*.gal|XML Weights|*.xml"
        fileDialog = wx.FileDialog(self,defaultDir=pathHint,message="Choose Weights File",wildcard=filter)
        result = fileDialog.ShowModal()
        if result == wx.ID_OK:
            path = fileDialog.GetPath()
            self.model.addKWeightsFile(path=path)
            self.KernelWeightsWarning(path)
        else:
            print "canceled"
    def KernelWeightsWarning(self,path):
        if self.model.checkKW(path):
            pass
        else:
            dialog = wx.MessageDialog(self,"Please use distance based weights with sufficient # of neighbors (see manual).","Kernel Weights Warning:",wx.OK|wx.ICON_ERROR)
            dialog.ShowModal()
            
        
    def DataOpenButtonClick(self,evt):
        filter = "dBase file (*.dbf)|*.dbf|Comma Seperated Values (*.csv;*.txt)|*.csv;*.txt"
        fileDialog = wx.FileDialog(self,message="Choose File",wildcard=filter)
        result = fileDialog.ShowModal()
        if result == wx.ID_OK:
            path = fileDialog.GetPath()
            self.setDataFile(path)
        else:
            print "canceled"
    #def setIDVar(self,e):
    #    self.model.setIDVar(self.IDVAR.GetSelection()) # -1 for the "Use Record Order Option"
    def setDataFile(self,path,passive=False):
        #if not path:
        #    path = self.model.dataFiles[self.DATAFILE.GetSelection()]
        self.model.setDataFile(path,passive)
        self.newVarSelector()

    def populate(self,model):
        m = self.model
        dfile = m.getDataFile()
        #if self.DATAFILE.FindString(dfile) == wx.NOT_FOUND and dfile:
        #    self.DATAFILE.Append(dfile)
        self.DATAFILE.SetValue(dfile)
        #vars = m.getVariables()
        #vars.insert(0,'Use Record Order')
        #if vars:
        #    self.IDVAR.Clear()
        #    self.IDVAR.AppendItems(vars)
        #    self.IDVAR.SetSelection(m.getIDVar()) #+1 for the "Use Record Order" option
        #vars.pop(0)
        self.setModelType(m.getModelType())
        mw = m.getMWeightsFiles()
        #if mw:
        self.MWeights_ListBox.SetItems(mw)
        kw = m.getKWeightsFiles()
        #if kw:
        self.KWeights_ListBox.SetItems(kw)
        self.setSpec(m.getSpec())
        self.setTitle()
        self.able()
    def notyet(self,e=None):
        print "Sorry, this feature is not yet implemented."
    def verbose(self,model):
        print self.model.data
    def run(self,evt):
        #fname = self.model.data['fname']
        #suggestion = os.path.split(fname)[1].split('.')[0]+'.txt'
        #fileDialog = wx.FileDialog(self,defaultFile=suggestion,message="Save Results As...",wildcard="*.txt",style=wx.SAVE+wx.OVERWRITE_PROMPT)
        #result = fileDialog.ShowModal()
        #if result == wx.ID_OK:
        #    path = fileDialog.GetPath()
        #    if not '.' in path:
        #        path += '.txt'
        try:
            pos = self.textFrame.Text.GetLastPosition()
            result = self.model.run(self.textFrame)
            if not result: #model.verify failed
                cause = self.model.verify()[1]
                dialog = wx.MessageDialog(self,cause,"Model Error",wx.OK|wx.ICON_ERROR)
                dialog.ShowModal()
                return False
            self.results.extend(result)
            #self.textFrame.write(50*'='+'\n')
            #self.resultText.seek(0)
            #self.textFrame.Text.SetValue(self.resultText.read())
            self.textFrame.Text.SetInsertionPoint(pos)
            #self.textFrame.SetModified(True)
            self.textFrame.Show()
            self.textFrame.Raise()
            #dialog = wx.MessageDialog(self,"Your model ran successfully! The results have been saved in,\n%s"%path,"Success",wx.OK|wx.ICON_INFORMATION)
            #dialog.ShowModal()
        except KeyError:
            #commonly caused by no or incorrectly set ID Var.
            dialog = wx.MessageDialog(self,"Please check that the ID variable of your weights file matches your dbf file or create a new weights file in GeoDaWeights.","The model failed to run. ",wx.OK|wx.ICON_ERROR)
            dialog.ShowModal()
            raise
        except AttributeError:
            #commonly caused by no weights.
            if self.model.data['modelType']['error']['hac'] and not self.model.data['kWeights']:
                dialog = wx.MessageDialog(self,"Kernel weights are required for HAC Standard Errors, please specify your kernel weights and try again.","The model failed to run.",wx.OK|wx.ICON_ERROR)
                dialog.ShowModal()
            else: 
                dialog = wx.MessageDialog(self,"Please be sure that you have specified your weights files correctly.","The model failed to run. ",wx.OK|wx.ICON_ERROR)
                dialog.ShowModal()
            raise
        except:
            dialog = wx.MessageDialog(self,"The model failed to run. Please Try Again.","Error",wx.OK|wx.ICON_ERROR)
            dialog.ShowModal()
            raise
        #else:
        #    print "canceled"
