#system
import os
import math
#3rd Party
import wx
import pysal
#local
from geodaspace import remapEvtsToDispatcher
from view import xrcDIALOGWEIGHTS
from model import weightsModel
from geodaspace import DEBUG
#CONSTANTS
ENABLE_CONTIGUITY_WEIGHTS = 1 # 0b00000001
ENABLE_DISTANCE_WEIGHTS = 2   # 0b00000010
ENABLE_KERNEL_WEIGTHS = 4     # 0b00000100
WEIGHTS_DEFAULT_STYLE = ENABLE_CONTIGUITY_WEIGHTS|ENABLE_DISTANCE_WEIGHTS|ENABLE_KERNEL_WEIGTHS

class weightsDialog(xrcDIALOGWEIGHTS):
    """
    Weights Dialog -- Displaces a Dialog for creating weights

    Display using ShowModal, which will return, wx.ID_OK or wx.ID_CANCEL

    Parameters
    ----------
    parent -- wxWindow -- A parent to the dialog, optional.
    requireSave -- bool -- If true the user will be prompted to save their weights objects to files.
    style -- bitmap -- Valid styles are, ENABLE_CONTIGUITY_WEIGHTS, ENABLE_DISTANCE_WEIGHTS, ENABLE_KERNEL_WEIGTHS or WEIGHTS_DEFAULT_STYLE
                        WEIGHTS_DEFAULT_STYLE is equal to ENABLE_CONTIGUITY_WEIGHTS|ENABLE_DISTANCE_WEIGHTS|ENABLE_KERNEL_WEIGTHS

    Methods
    _______
    GetW -- returns a W object.
    HasW -- returns a bool.
    """
    def __init__(self, parent = None, requireSave = True, style = WEIGHTS_DEFAULT_STYLE):
        remapEvtsToDispatcher(self, self.evtDispatch)
        xrcDIALOGWEIGHTS.__init__(self, parent)
        self.ContiguityPage = self.ContiguityPanel, self.weightsNotebook.GetPageText(0)
        self.DistancePage = self.DistancePanel, self.weightsNotebook.GetPageText(1)
        self.KernelPage = self.KernelPanel, self.weightsNotebook.GetPageText(2)

        self.CutoffText.Bind(wx.EVT_CHAR,self.isdigit)
        openID = wx.NewId()
        aTable = wx.AcceleratorTable([(wx.ACCEL_CMD, ord("O"), openID)])
        self.SetAcceleratorTable(aTable)
        self.Bind(wx.EVT_MENU, self.OnButton_OpenShape, id=openID)

        self.requireSave = requireSave
        self.orig_style = style
        self.hidden_frame = wx.Frame(self)
        
        self.model = weightsModel()
        self.model.addListener(self.update)

        self.dispatch = d = {}
        d['OpenShape'] = self.input
        d['InputShapeChoice'] = self.input
        d['inShp'] = self.input
        d['inShps'] = self.input
        d['IdvarChoice'] = self.idVar
        d['idVar'] = self.idVar
        d['CreateButton'] = self.run
        d['ThresholdSlider'] = self.threshold
        d['CutoffText'] = self.threshold
        d['OnClose'] = self.closeEvt
        d['CloseButton'] = self.closeEvt
        self._W = None
        self.update_style(style)
        self.update()
    def evtDispatch(self,evtName, evt):
        evtName,widgetName = evtName.rsplit('_',1)
        if widgetName in self.dispatch:
            self.dispatch[widgetName](evtName,evt)
        else:
            if DEBUG: print "not implemented:", evtName, widgetName
    def update(self,tag=False):
        #if DEBUG: print "CONTROL... updating tag:",tag
        if tag:
            if tag in self.dispatch:
                self.dispatch[tag](value=self.model.getByTag(tag))
            else:
                if DEBUG: print "Warning: %s, has not been implemented"%tag
        else:
            for key,value in self.model:
                if key in self.dispatch:
                    self.dispatch[key](value=value)
                else:
                    if DEBUG: print "Warning: %s, has not been implemented"%key
        self.able()
    def able(self):
        """ Enable/Disable GUI Elements based on the state of the model """
        if self.model.shapes == None:
            self.weightsNotebook.Disable()
        else:
            self.weightsNotebook.Enable()
    def update_style(self, flags):
        #clear the existing pages.
        for pid in xrange(self.weightsNotebook.GetPageCount()-1,-1,-1):
            self.weightsNotebook.GetPage(pid).Reparent(self.hidden_frame)
            self.weightsNotebook.RemovePage(pid) #Note, Remove do not Delete!
        if flags & ENABLE_CONTIGUITY_WEIGHTS:
            self.ContiguityPanel.Reparent(self.weightsNotebook)
            self.weightsNotebook.AddPage(*self.ContiguityPage)
        if flags & ENABLE_DISTANCE_WEIGHTS:
            self.DistancePanel.Reparent(self.weightsNotebook)
            self.weightsNotebook.AddPage(*self.DistancePage)
        if flags & ENABLE_KERNEL_WEIGTHS:
            self.KernelPanel.Reparent(self.weightsNotebook)
            self.weightsNotebook.AddPage(*self.KernelPage)
        self.weightsNotebook.SetSelection(0)
        self.cur_style = flags
    def save(self):
        """ Prompt the user to save the weights file """
        if not self.HasW():
            return self.Warn("No weights object has been created yet.")
        try:
            w = self.GetW()
            if hasattr(w,'meta'):
                if w.meta['method'] == 'adaptive kernel':
                    filter = "Kernel Weights File (*.kwt)|*.kwt"
                    exts = {0:'.kwt'}
                elif w.meta['method'] == 'contiguity':
                    filter = "Weights File (*.gal)|*.gal"
                    exts = {0:'.gal'}
                else:
                    filter = "Weights File (*.gwt)|*.gwt"
                    exts = {0:'.gwt'}
            else:
                filter = "Weights File (*.gal)|*.gal|Weights File (*.gwt)|*.gwt"
                exts = {0:'.gal',1:'.gwt'}
            pathHint,filename = os.path.split(self.model.inShps[self.model.inShp])
            filename = filename[:-4]
            
            fileDialog = wx.FileDialog(self,defaultDir=pathHint,defaultFile=filename,
                                        message="Choose File",wildcard=filter,
                                        style=wx.SAVE+wx.OVERWRITE_PROMPT)
            result = fileDialog.ShowModal()
            if result == wx.ID_OK:
                path = fileDialog.GetPath()
                ext = [fileDialog.GetFilterIndex()]
                if not path.endswith(ext):
                    path = path+ext
                o = pysal.open(path, 'w')
                if ext == '.gwt':
                    try:
                        o.shpName = filename+'.shp'
                        o.varName = self.model.vars[self.model.idVar]
                    except:
                        self.warn("Could not set meta data of the GWT file. Will continue without it.")
                        if DEBUG: raise
                o.write(self.GetW())
                o.close()
                return True
            else:
                res = wx.MessageDialog(self, "Are you sure you don't want to save your weights object to a file?\nUnsaved weights object may be lost.", "Warning", style=wx.YES_NO|wx.NO_DEFAULT).ShowModal()
                if res == wx.ID_YES:
                    return False
                else:
                    return self.save()
        except:
            self.warn("An unknown error occurred while trying to save the weights object.")
            if DEBUG: raise
            return False

    def input(self, evtName=None, evt=None, value=None):
        """ Handles all events related to input files.  """
        if DEBUG: print "input:", evtName, evt, value
        if evt: #Event cause by change in GUI
            path = ''
            if evt.EventType in (wx.EVT_BUTTON.typeId, wx.EVT_MENU.typeId): #the open button
                filter = "Shape File (*.shp)|*.shp"
                fileDialog = wx.FileDialog(self,message="Choose File",wildcard=filter)
                result = fileDialog.ShowModal()
                if result == wx.ID_OK:
                    path = fileDialog.GetPath()
                else:
                    if DEBUG: print "in shape: canceled"
            elif evt.EventType == wx.EVT_CHOICE.typeId: # The drop down
                path = self.InputShapeChoice.GetSelection()
            if not path == '': self.model.inShp = path
        elif not value == None: #Value changed in the Model, update the GUI
            if type(value) == int:
                self.InputShapeChoice.SetSelection(value)
            elif value:
                self.InputShapeChoice.Clear()
                if '' in value:
                    value.remove('')
                value = [os.path.split(path)[-1] for path in value]
                self.InputShapeChoice.AppendItems(value)
                if self.model.inShp != '':
                    self.InputShapeChoice.SetSelection(self.model.inShp)
            self.IdvarChoice.Clear()
            self.IdvarChoice.AppendItems(self.model.vars)
            # Set Slider Values...            
            if self.model.bbox_diag > 0:
                max_dist = self.model.bbox_diag
                rec_dist = self.model.knn1_dist
                self.ThresholdSlider.SetValue(int(math.ceil((rec_dist/max_dist) * self.ThresholdSlider.GetMax())))
                self.threshold()
            if self.model.shapes:
                n = len(self.model.shapes)
                # Min k for Kernel weights is the cube root of the number of observations.
                min_k = int(math.ceil(n**(1/3.0)))
                self.KNumNeighSpin.SetRange(min_k,n)
                self.KNumNeighSpin.SetValue(min_k)
                self.NumNeighSpin.SetRange(1,n)
                if self.model.shapes.type == pysal.cg.Point:
                    if ENABLE_CONTIGUITY_WEIGHTS&self.orig_style and self.cur_style&ENABLE_CONTIGUITY_WEIGHTS:
                        self.update_style(self.cur_style^ENABLE_CONTIGUITY_WEIGHTS)
                elif self.model.shapes.type == pysal.cg.Polygon:
                    if ENABLE_CONTIGUITY_WEIGHTS&self.orig_style:
                        self.update_style(self.cur_style|ENABLE_CONTIGUITY_WEIGHTS)

    def isdigit(self,evt):
        """easy validator for textCtrl
            Simply consumes the EVT_CHAR if it doesn't like the change, otherwise the evt is skipped and the TextCtrl receives it.
        """
        key = evt.GetKeyCode() 
        try: character = chr(key) 
        except ValueError: character = "" # arrow keys will throw this error 

        acceptable_characters = "1234567890"
        # 13 = enter, 314 & 316 = arrows, 8 = backspace, 127 = del 
        if character in acceptable_characters or key == 13 or key == 314 or key == 316 or key == 8 or key == 127:
            evt.Skip() 
            return 
        elif character == '.':
            if '.' in self.CutoffText.GetValue(): #already a . in the number
                return False
            else:
                evt.Skip()
        else: 
            return False 

    def threshold(self, evtName=None, evt=None, value=None):
        if evtName == 'OnText':
            #user is typing...
            if self.CutoffText.IsModified():
                val = float(self.CutoffText.GetValue())
                x = int(math.ceil((val/self.model.bbox_diag) * self.ThresholdSlider.GetMax()))
                x = x if x < self.ThresholdSlider.GetMax() else self.ThresholdSlider.GetMax()
                self.ThresholdSlider.SetValue(x)
        else:
            if self.model.bbox_diag:
                val = self.ThresholdSlider.GetValue()
                pct = val / float(self.ThresholdSlider.GetMax())
                self.CutoffText.ChangeValue(str(self.model.bbox_diag * pct))
    def idVar(self, evtName=None, evt=None, value=None):
        if evtName == "OnChoice":
            self.model.idVar = self.IdvarChoice.GetSelection()
        elif value != None:
            self.IdvarChoice.SetSelection(value)
    def warn(self,msg):
        wx.MessageDialog(self, msg, "Warning", style=wx.OK|wx.ICON_HAND).ShowModal()
    def GetW(self):
        """ Returns the W object created by the user """
        return self._W
    def SetW(self,W):
        self._W = W
    def HasW(self):
        if self._W: return True
        else: return False
    def run_contiguity(self,sfile,var):
        """ Invoked by main run method. """
        if self.model.shapes.type != pysal.cg.Polygon:
            return self.warn("The selected shapefile does not contain polygons and contiguity weights can only be computed on polygons.")
        order = self.ContiguityOrderSpin.GetValue()
        include_lower = self.ContiguityIncludeLowerCheck.GetValue()
        if self.RadioRook.GetValue():
            cont_type = 'rook'
        elif self.RadioQueen.GetValue():
            cont_type = 'queen'
        func = {'rook':pysal.rook_from_shapefile, 'queen':pysal.queen_from_shapefile}[cont_type]
        W = func(sfile,var)
        if order > 1:
            W_orig = W
            W = W.higher_order(order)
            if include_lower:
                for o in xrange(order-1,1,-1):
                    W = pysal.weights.w_union(W_orig.higher_order(o), W)
                W = pysal.weights.w_union(W, W_orig)
        W.meta = {'shape file':sfile,
                  'id variable':var,
                  'method':'contiguity',
                  'method options':cont_type}
        self.SetW(W)
    def run_distance(self,sfile,var):
        """ Invoked by main run method. """
        if self.model.shapes.type == pysal.cg.Polygon:
            self.warn("The selected shapefile contains polygons and distance weights can only be computed on points. "+\
                      "The centroids of the specified polygons will be used instead.")
        elif self.model.shapes.type != pysal.cg.Point:
            return self.warn("The selected shapefile does not contain points and contiguity weights can only be computed on points.")
        try: cutoff = float(self.CutoffText.GetValue())
        except: return self.warn("The cut-off point is not valid.")
        if self.ThresholdRadio.GetValue():
            print "Threshold on %s, ids=%r, cutoff=%f"%(sfile,var,cutoff)
            W = pysal.threshold_binaryW_from_shapefile(sfile, cutoff, idVariable=var)
            W.meta = {'shape file':sfile,
                      'id variable':var,
                      'method':'distance',
                      'method options':['Threshold',cutoff]}
            self.SetW(W)
        elif self.KnnRadio.GetValue():
            k = int(self.NumNeighSpin.GetValue())
            print "Knn on %s, ids=%r, k=%d"%(sfile,var,k)
            W = pysal.knnW_from_shapefile(sfile, k=k, idVariable=var)
            W.meta = {'shape file':sfile,
                      'id variable':var,
                      'method':'distance',
                      'method options':['KNN',k]}
            self.SetW(W)
        elif self.InverseRadio.GetValue():
            power = int(self.PowerSpin.GetValue())
            print "Inverse on %s, ids=%r, cutoff=%f, power=%d"%(sfile,var,cutoff,power)
            W = pysal.threshold_continuousW_from_shapefile(sfile, cutoff, alpha=-1*power, idVariable=var)
            W.meta = {'shape file':sfile,
                      'id variable':var,
                      'method':'distance',
                      'method options':['Inverse',cutoff,power]}
            self.SetW(W)
    def run_adaptive_kernel(self,sfile,var):
        """ Invoked by main run method. """
        if self.model.shapes.type == pysal.cg.Polygon:
            self.warn("The selected shapefile contains polygons and kernel weights can only be computed on points. "+\
                      "The centroids of the specified polygons will be used instead.")
        elif self.model.shapes.type != pysal.cg.Point:
            return self.warn("The selected shapefile does not contain points and kernel weights can only be computed on points.")
        kern = ['uniform','triangular','quadratic','quartic','gaussian'][self.KFuncChoice.GetSelection()]
        k = int(self.KNumNeighSpin.GetValue())
        print "Kernel on %s, k=%d, ids=%r, kernel=%s"%(sfile, k, var, kern)
        W = pysal.adaptive_kernelW_from_shapefile(sfile, k=k, function=kern, idVariable=var)
        W.meta = {'shape file':sfile,
                  'id variable':var,
                  'method':'adaptive kernel',
                  'method options':[kern,k]}
        self.SetW(W)
    def run(self, evtName=None, evt=None, value=None):
        if self.model.shapes != None:
            sfile = self.model.inShps[self.model.inShp]
            method = self.weightsNotebook.GetPageText(self.weightsNotebook.GetSelection())
            var = None
            if self.model.idVar != None:
                var = self.model.vars[self.model.idVar]
            if method == 'Contiguity':
                self.run_contiguity(sfile,var)
            elif method == 'Distance':
                self.run_distance(sfile,var)
            elif method == 'Adaptive Kernel':
                self.run_adaptive_kernel(sfile,var)
        if self.HasW():
            if self.requireSave:
                self.save()
            self.close()

    def close(self,ret_val=wx.ID_OK):
        if self.IsModal():
            self.EndModal(ret_val)
        else:
            self.Hide()
    def OnClose(self, evt):
        self.close(ret_val=wx.ID_CANCEL)
    def closeEvt(self, evtName=None, evt=None, value=None):
        self.close(ret_val=wx.ID_CANCEL)
    def closeEvt(self, evtName=None, evt=None, value=None):
        self.close(ret_val=wx.ID_CANCEL)
    def closeEvt(self, evtName=None, evt=None, value=None):
        self.close(ret_val=wx.ID_CANCEL)
