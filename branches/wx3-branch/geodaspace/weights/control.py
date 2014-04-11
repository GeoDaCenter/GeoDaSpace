# system
import os
import math
import sys
# 3rd Party
import wx
import pysal
import pysal.contrib.weights_viewer.weights_viewer as weights_viewer
# local
from geodaspace import remapEvtsToDispatcher
from geodaspace import DEBUG
from view import xrcDIALOGWEIGHTS, xrcAddIDVar, xrcweightsProperties
from model import weightsModel, DISTANCE_METRICS
# CONSTANTS
ENABLE_CONTIGUITY_WEIGHTS = 1  # 0b00000001
ENABLE_DISTANCE_WEIGHTS = 2   # 0b00000010
ENABLE_KERNEL_WEIGHTS = 4     # 0b00000100
WEIGHTS_DEFAULT_STYLE = ENABLE_CONTIGUITY_WEIGHTS | ENABLE_DISTANCE_WEIGHTS |\
    ENABLE_KERNEL_WEIGHTS

WEIGHT_TYPES_FILTER = "ArcGIS DBF files (.dbf)|*.dbf|ArcGIS SWM files \
        (*.swm)|*.swm|ArcGIS Text files (*.txt)|*.txt|DAT files \
        (*.dat)|*.dat|GAL files (*.gal)|*.gal|GeoBUGS Text files \
        (*.)|*.|GWT files (*.gwt)|*.gwt|KWT files (*.kwt)|*.kwt|MatLab files\
        (*.mat)|*.mat|MatrixMarket files (*.mtx)|*.mtx|STATA Text files\
        (*.txt)|*.txt"

WEIGHT_FILTER_TO_HANDLER = {0: 'arcgis_dbf',
                            1: None,
                            2: 'arcgis_text',
                            3: None,
                            4: None,
                            5: 'geobugs_text',
                            6: None,
                            7: None,
                            8: None,
                            9: None,
                            10: 'stata_text',
                            11: None}

VALID_TRANSFORMS = [
    "B: Binary",
    "R: Row-standardization (global sum=n)",
    "D: Double-standardization (global sum=1)",
    "V: Variance stabilizing",
    "O: Restore original transformation (from instantiation)"]

TRANSFORM_LOOKUP = dict([(x.split(':')[0], i)
                         for i, x in enumerate(VALID_TRANSFORMS)])


class weightsPropertiesDialog(xrcweightsProperties):
    """
    Weights Properties Dialog --
    Displays a Dialog viewing and editor properties of a W obj.

    Display using ShowModal, which requires a list of W objects
    and an optional selection, returns wx.ID_CLOSE

    Parameters
    ----------
    parent -- wxWindow -- A parent to the dialog, optional.

    """
    def __init__(self, parent=None):
        xrcweightsProperties.__init__(self, parent)
        self.Bind(wx.EVT_CLOSE, self.onClose)
        self.transformChoice.AppendItems(VALID_TRANSFORMS)

    def onClose(self, evt):
        self.MakeModal(False)
        self.Hide()

    def ShowModal(self, w_objs, selection=-1):
        """
        Display the Dialog and add the id var.
        """
        self.w_objs = w_objs
        self.selectWChoice.Clear()
        self.selectWChoice.AppendItems([w.name for w in w_objs])
        selection = 0 if selection < 0 else selection
        self.selectWChoice.SetSelection(selection)
        self.OnChoice_selectWChoice(None)
        # names = []
        # if os.path.exists(dbf_path):
        #    self.db = pysal.open(dbf_path,'r')
        #    self.existingVarsListBox.Clear()
        #    self.existingVarsListBox.InsertItems(self.db.header,0)
        #    self.idVarName.SetValue('POLY_ID')
        #    return xrcAddIDVar.ShowModal(self)
        # else:
        #    raise ValueError, "Invalid DBF File"
        xrcweightsProperties.Show(self)
        xrcweightsProperties.MakeModal(self, True)

    def OnChoice_selectWChoice(self, evt):
        w = self.w_objs[self.selectWChoice.GetSelection()]
        self.idListChoice.Clear()
        self.idListChoice.AppendItems(map(str, w.w.id_order))
        self.idListChoice.SetSelection(0)
        self.update()

    def OnChoice_idListChoice(self, evt):
        w = self.w_objs[self.selectWChoice.GetSelection()]
        id = w.w.id_order[self.idListChoice.GetSelection()]
        self.weightsTC.SetValue(str(w.w[id]))

    def OnChoice_transformChoice(self, evt):
        w = self.w_objs[self.selectWChoice.GetSelection()]
        sel = self.transformChoice.GetSelection()
        T = VALID_TRANSFORMS[sel].split(':')[0]
        w.w.transform = T
        self.update()

    def OnButton_viewerButton(self, evt):
        w = self.w_objs[self.selectWChoice.GetSelection()]
        print w.shapefile_hint
        if os.path.exists(w.shapefile_hint):
            geo = w.shapefile_hint
        else:
            fileDialog = wx.FileDialog(self, message="Please locate: %s" %
                                       w.shapefile_hint,
                                       wildcard="Shape File (*.shp)|*.shp")
            result = fileDialog.ShowModal()
            if result == wx.ID_OK:
                geo = fileDialog.GetPath()
            else:
                return
        wm = weights_viewer.WeightsMapFrame(
            self, geo=geo, w=w.w,
            style=wx.DEFAULT_FRAME_STYLE | wx.FRAME_FLOAT_ON_PARENT)
        wm.Show(True)

    def OnButton_closeButton(self, evt):
        xrcweightsProperties.MakeModal(self, False)
        xrcweightsProperties.Hide(self)

    def update(self):
        w = self.w_objs[self.selectWChoice.GetSelection()]
        self.nameTC.SetValue(w.name)
        self.transformChoice.SetSelection(TRANSFORM_LOOKUP[w.w.transform])
        islands = w.w.islands
        if not islands:
            self.islandsTC.SetValue("No Islands")
        else:
            self.islandsTC.SetValue(str(islands))
        id = w.w.id_order[self.idListChoice.GetSelection()]
        self.weightsTC.SetValue(str(w.w[id]))
        self.cardinalitiesTC.SetValue(str(w.w.cardinalities))
        self.idsTC.SetValue(str(w.w.id_order))
        self.histogramTC.SetValue(str(w.w.histogram))


class idVarDialog(xrcAddIDVar):
    """
    Add ID Variable Dialog --
    Displays a Dialog for adding a unique id to a DBF File.

    Display using ShowModal, which requires a DBF File Path
    and returns wx.ID_OK or wx.ID_CANCEL

    Parameters
    ----------
    parent -- wxWindow -- A parent to the dialog, optional.
    """
    def __init__(self, parent=None):
        xrcAddIDVar.__init__(self, parent)

    def OnButton_save(self, evt):
        self.EndModal(wx.ID_OK)

    def OnButton_cancel(self, evt):
        xrcAddIDVar.EndModal(self, wx.ID_CANCEL)

    def ShowModal(self, dbf_path):
        """
        Display the Dialog and add the id var.
        """
        self.db = None
        self.dbf_path = dbf_path
        if os.path.exists(dbf_path):
            self.db = pysal.open(dbf_path, 'r')
            self.existingVarsListBox.Clear()
            self.existingVarsListBox.InsertItems(self.db.header, 0)
            self.idVarName.SetValue('POLY_ID')
            return xrcAddIDVar.ShowModal(self)
        else:
            raise ValueError("Invalid DBF File")

    def verify_name(self, name):
        try:
            assert len(name) > 0
            assert len(name) < 11
            assert name[0].isalpha()
            assert all([x.isalnum() or x == '_' for x in name])
            return True
        except AssertionError:
            wx.MessageDialog(
                self, "Error: \"%s\" is an invalid field name.\
                A valid field name is between one and ten characters long.\
                The first character must be alphabetic, and the remaining \
                characters can be either alphanumeric or underscores." %
                name, "Error", style=wx.ICON_HAND).ShowModal()
            return False

    def AddNewIDVar(self, name):
        header = self.db.header
        spec = self.db.field_spec
        n = len(self.db)
        header.insert(0, name)
        spec.insert(0, ('N', len(str(n)), 0))
        new_rows = [[i + 1] + row for i, row in enumerate(self.db)]
        self.db.close()
        new_db = pysal.open(self.dbf_path, 'w')
        new_db.header = header
        new_db.field_spec = spec
        for row in new_rows:
            new_db.write(row)
        new_db.close()

    def EndModal(self, ret_code):
        name = self.idVarName.GetValue().upper().encode('ascii')
        if not self.verify_name(name):
            return
        if name in self.db.header:
            wx.MessageDialog(
                self, "Error: Field name \"%s\" already exists in the DBF\
                file.\nPlease choose a different name." % (
                name,), style=wx.ICON_HAND).ShowModal()
            return
        shpName = os.path.splitext(os.path.basename(self.dbf_path))[0]

        dlg = wx.MessageDialog(
            self, "Are you sure you want to add the new id variable to the DBF\
            file associated with the chosen input SHP file \"%s\"" % (
            shpName,), "Add id variable to DBF file?",
            style=wx.ICON_HAND | wx.YES_NO)
        if dlg.ShowModal() == wx.ID_YES:
            self.AddNewIDVar(name)
        else:
            return
        return xrcAddIDVar.EndModal(self, ret_code)


class weightsDialog(xrcDIALOGWEIGHTS):
    """
    Weights Dialog -- Displays a Dialog for creating weights

    Display using ShowModal, which will return, wx.ID_OK or wx.ID_CANCEL

    Parameters
    ----------
    parent : wxWindow : A parent to the dialog, optional.

    requireSave : bool :
    If true the user will be prompted to save their weights objects to files.

    style : bitmap :
    Valid styles are ENABLE_CONTIGUITY_WEIGHTS, ENABLE_DISTANCE_WEIGHTS,
    ENABLE_KERNEL_WEIGHTS or WEIGHTS_DEFAULT_STYLE
    WEIGHTS_DEFAULT_STYLE is equal to
    ENABLE_CONTIGUITY_WEIGHTS|ENABLE_DISTANCE_WEIGHTS|ENABLE_KERNEL_WEIGHTS

    Methods
    _______
    GetW -- returns a W object.
    HasW -- returns a bool.
    """
    def __init__(self, parent=None, requireSave=True,
                 style=WEIGHTS_DEFAULT_STYLE):
        remapEvtsToDispatcher(self, self.evtDispatch)
        xrcDIALOGWEIGHTS.__init__(self, parent)
        self.ContiguityPage = self.ContiguityPanel,\
            self.weightsNotebook.GetPageText(0)
        self.DistancePage = self.DistancePanel,\
            self.weightsNotebook.GetPageText(1)
        self.KernelPage = self.KernelPanel, self.weightsNotebook.GetPageText(2)

        self.CutoffText.Bind(wx.EVT_CHAR, self.isdigit)
        openID = wx.NewId()
        aTable = wx.AcceleratorTable([(wx.ACCEL_CMD, ord("O"), openID)])
        self.SetAcceleratorTable(aTable)
        self.Bind(wx.EVT_MENU, self.OnButton_OpenShape, id=openID)

        self.requireSave = requireSave
        self.orig_style = style
        self.hidden_frame = wx.Frame(self)

        self.model = weightsModel()
        self.model.addListener(self.update)

        self.DdistMethodChoice.SetItems(DISTANCE_METRICS)
        self.KdistMethodChoice.SetItems(DISTANCE_METRICS)

        self.dispatch = d = {}
        d['OpenShape'] = self.input
        d['InputShapeChoice'] = self.input
        d['inShp'] = self.input
        d['inShps'] = self.input
        d['IdvarChoice'] = self.idVar
        d['idVar'] = self.idVar
        d['addIDVar'] = self.idVar
        d['CreateButton'] = self.run
        d['ThresholdSlider'] = self.threshold
        d['ThresholdSlider2'] = self.threshold2
        d['CutoffText'] = self.threshold
        d['CutoffText2'] = self.threshold2
        d['OnClose'] = self.closeEvt
        d['CloseButton'] = self.closeEvt
        d['DdistMethodChoice'] = self.distMeth
        d['KdistMethodChoice'] = self.distMeth
        d['distMethod'] = self.distMeth
        self._W = None
        self.update_style(style)
        self.update()

    def evtDispatch(self, evtName, evt):
        evtName, widgetName = evtName.rsplit('_', 1)
        if widgetName in self.dispatch:
            self.dispatch[widgetName](evtName, evt)
        else:
            if DEBUG:
                print "not implemented:", evtName, widgetName

    def update(self, tag=False):
        # if DEBUG: print "CONTROL... updating tag:",tag
        if tag:
            if tag in self.dispatch:
                self.dispatch[tag](value=self.model.getByTag(tag))
            else:
                if DEBUG:
                    print "Warning: %s, has not been implemented" % tag
        else:
            for key, value in self.model:
                if key in self.dispatch:
                    self.dispatch[key](value=value)
                else:
                    if DEBUG:
                        print "Warning: %s, has not been implemented" % key
        self.able()

    def able(self):
        """ Enable/Disable GUI Elements based on the state of the model """
        if self.model.shapes is None:
            self.weightsNotebook.Disable()
            self.IdvarChoice.Disable()
            self.addIDVar.Disable()
        else:
            self.IdvarChoice.Enable()
            self.addIDVar.Enable()
        if self.model.idVar is not None:
            self.weightsNotebook.Enable()
        else:
            self.weightsNotebook.Disable()

    def update_style(self, flags):
        # clear the existing pages.
        for pid in xrange(self.weightsNotebook.GetPageCount() - 1, - 1, - 1):
            self.weightsNotebook.GetPage(pid).Reparent(self.hidden_frame)
            self.weightsNotebook.RemovePage(pid)  # Note, Remove do not Delete!
        if flags & ENABLE_CONTIGUITY_WEIGHTS:
            self.ContiguityPanel.Reparent(self.weightsNotebook)
            self.weightsNotebook.AddPage(*self.ContiguityPage)
        if flags & ENABLE_DISTANCE_WEIGHTS:
            self.DistancePanel.Reparent(self.weightsNotebook)
            self.weightsNotebook.AddPage(*self.DistancePage)
        if flags & ENABLE_KERNEL_WEIGHTS:
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
            exts = WEIGHT_TYPES_FILTER.split('|')
            exts = [exts[i + 1] for i in range(0, len(exts), 2)]
            if hasattr(w, 'meta'):
                if w.meta['method'] == 'adaptive kernel':
                    # filter = "Kernel Weights File (*.kwt)|*.kwt"
                    # exts = {0:'.kwt'}
                    suggested = exts.index('*.kwt')
                elif w.meta['method'] == 'contiguity':
                    # filter = "Weights File (*.gal)|*.gal"
                    # exts = {0:'.gal'}
                    suggested = exts.index('*.gal')
                else:
                    # filter = "Weights File (*.gwt)|*.gwt"
                    # exts = {0:'.gwt'}
                    suggested = exts.index('*.gwt')
            else:
                #filter="Weights File (*.gal)|*.gal|Weights File (*.gwt)|*.gwt"
                # exts = {0:'.gal',1:'.gwt'}
                suggested = exts.index('*.gal')
            if hasattr(w, 'meta') and 'shape file' in w.meta:
                pathHint, filename = os.path.split(w.meta['shape file'])
            elif self.model.inShp != '':
                pathHint, filename = os.path.split(
                    self.model.inShps[self.model.inShp])
            else:
                pathHint, filename = '', ''
            filename = filename[:-4]

            fileDialog = wx.FileDialog(
                self, defaultDir=pathHint, defaultFile=filename,
                message="Choose File", wildcard=WEIGHT_TYPES_FILTER,
                style=wx.SAVE + wx.OVERWRITE_PROMPT)
            fileDialog.SetFilterIndex(suggested)
            result = fileDialog.ShowModal()
            if result == wx.ID_OK:
                path = fileDialog.GetPath()
                ext = '.' + exts[fileDialog.GetFilterIndex()].split('.')[1]
                handler = WEIGHT_FILTER_TO_HANDLER[fileDialog.GetFilterIndex()]
                if not path.endswith(ext):
                    path = path + ext
                o = pysal.open(path, 'w', handler)
                if ext in ['.gwt', '.kwt']:
                    try:
                        o.shpName = filename + '.shp'
                        if hasattr(w, 'meta') and 'id variable' in w.meta:
                            o.varName = w.meta['id variable']
                        else:
                            o.varName = self.model.vars[self.model.idVar]
                    except:
                        self.warn(
                            "Could not set meta data of the GWT file. \
                            Will continue without it.")
                        if DEBUG:
                            raise
                o.write(w)
                o.close()
                w.meta['savedAs'] = path
                return True
            else:
                res = wx.MessageDialog(
                    self, "This weights object will only be available in the \
                    current GeoDaSpace session", "Warning",
                    style=wx.OK | wx.ICON_HAND).ShowModal()
        except:
            self.warn(
                "An unknown error occurred while trying to save the \
                weights object.")
            if DEBUG:
                raise
            return False

    def input(self, evtName=None, evt=None, value=None):
        """ Handles all events related to input files.  """
        if DEBUG:
            print "input:", evtName, evt, value
        if evt:  # Event cause by change in GUI
            path = ''
            if evt.EventType in (wx.EVT_BUTTON.typeId, wx.EVT_MENU.typeId):
                # the open button
                filter = "Shape File (*.shp)|*.shp"
                fileDialog = wx.FileDialog(
                    self, message="Choose File", wildcard=filter)
                result = fileDialog.ShowModal()
                if result == wx.ID_OK:
                    path = fileDialog.GetPath()
                else:
                    if DEBUG:
                        print "in shape: canceled"
            elif evt.EventType == wx.EVT_CHOICE.typeId:  # The drop down
                path = self.InputShapeChoice.GetSelection()
            if not path == '':
                self.model.inShp = path
        elif not value is None:  # Value changed in the Model, update the GUI
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
            self.distance_hints()
            if self.model.shapes:
                n = len(self.model.shapes)
                # Min k for Kernel weights is the cube root of the number of
                # observations.
                min_k = int(math.ceil(n ** (1 / 3.0)))
                self.KNumNeighSpin.SetRange(min_k, n)
                self.KNumNeighSpin.SetValue(min_k)
                self.NumNeighSpin.SetRange(1, n)
                if self.model.shapes.type == pysal.cg.Point:
                    if ENABLE_CONTIGUITY_WEIGHTS & self.orig_style and \
                       self.cur_style & ENABLE_CONTIGUITY_WEIGHTS:
                        self.update_style(
                            self.cur_style ^ ENABLE_CONTIGUITY_WEIGHTS)
                elif self.model.shapes.type == pysal.cg.Polygon:
                    if ENABLE_CONTIGUITY_WEIGHTS & self.orig_style:
                        self.update_style(
                            self.cur_style | ENABLE_CONTIGUITY_WEIGHTS)

    def distance_hints(self, evtName=None, evt=None, value=None):
            # Set Slider Values...
            if self.model.bbox_diag > 0:
                max_dist = self.model.bbox_diag
                rec_dist = self.model.knn1_dist
                # self.ThresholdSlider.SetValue(int(math.ceil((
                    #rec_dist/max_dist) * self.ThresholdSlider.GetMax())))
                # self.ThresholdSlider2.SetValue(int(math.ceil(
                    #(rec_dist/max_dist)
                # * self.ThresholdSlider2.GetMax())))
                self.CutoffText.ChangeValue(str(self.model.knn1_dist))
                self.CutoffText.SetModified(True)
                self.CutoffText2.ChangeValue(str(self.model.knn1_dist))
                self.CutoffText2.SetModified(True)
                self.threshold(evtName='OnText')
                self.threshold2(evtName='OnText')

    def isdigit(self, evt):
        """easy validator for textCtrl
            Simply consumes the EVT_CHAR if it doesn't like the change,
            otherwise the evt is skipped and the TextCtrl receives it.
        """
        key = evt.GetKeyCode()
        try:
            character = chr(key)
        except ValueError:
            character = ""  # arrow keys will throw this error

        acceptable_characters = "1234567890"
        # 13 = enter, 314 & 316 = arrows, 8 = backspace, 127 = del
        if character in acceptable_characters or key == 13 or key == 314 or \
           key == 316 or key == 8 or key == 127:
            evt.Skip()
            return
        elif character == '.':
            if '.' in self.CutoffText.GetValue():  # already a . in the number
                return False
            else:
                evt.Skip()
        else:
            return False

    def threshold(self, evtName=None, evt=None, value=None):
        if evtName == 'OnText':
            # user is typing...
            if self.CutoffText.IsModified():
                val = float(self.CutoffText.GetValue())
                x = int(math.ceil((
                    val / self.model.bbox_diag) *
                    self.ThresholdSlider.GetMax()))
                x = x if x < self.ThresholdSlider.GetMax(
                ) else self.ThresholdSlider.GetMax()
                self.ThresholdSlider.SetValue(x)
        else:
            if self.model.bbox_diag:
                val = self.ThresholdSlider.GetValue()
                pct = val / float(self.ThresholdSlider.GetMax())
                self.CutoffText.ChangeValue(str(self.model.bbox_diag * pct))

    def threshold2(self, evtName=None, evt=None, value=None):
        if evtName == 'OnText':
            # user is typing...
            if self.CutoffText2.IsModified():
                val = float(self.CutoffText2.GetValue())
                x = int(math.ceil((
                    val / self.model.bbox_diag) *
                    self.ThresholdSlider2.GetMax()))
                x = x if x < self.ThresholdSlider2.GetMax(
                ) else self.ThresholdSlider2.GetMax()
                self.ThresholdSlider2.SetValue(x)
        else:
            if self.model.bbox_diag:
                val = self.ThresholdSlider2.GetValue()
                pct = val / float(self.ThresholdSlider2.GetMax())
                self.CutoffText2.ChangeValue(str(self.model.bbox_diag * pct))

    def idVar(self, evtName=None, evt=None, value=None):
        if evtName == "OnChoice":
            self.model.idVar = self.IdvarChoice.GetSelection()
        elif evtName == "OnButton":
            if idVarDialog(self).ShowModal(self.model.data_path) == wx.ID_OK:
                # trigger update of idVars list
                self.model.update('inShp')
                self.model.idVar = 0
        elif value is not None:
            self.IdvarChoice.SetSelection(value)

    def warn(self, msg):
        wx.MessageDialog(
            self, msg, "Warning", style=wx.OK | wx.ICON_HAND).ShowModal()

    def GetW(self):
        """ Returns the W object created by the user """
        return self._W

    def SetW(self, W):
        self._W = W

    def HasW(self):
        if self._W:
            return True
        else:
            return False

    def run_contiguity(self, sfile, var):
        """ Invoked by main run method. """
        if self.model.shapes.type != pysal.cg.Polygon:
            return self.warn("The selected shapefile does not contain polygons\
                             and contiguity weights can only be computed on \
                             polygons.")
        order = self.ContiguityOrderSpin.GetValue()
        include_lower = self.ContiguityIncludeLowerCheck.GetValue()
        if self.RadioRook.GetValue():
            cont_type = 'rook'
        elif self.RadioQueen.GetValue():
            cont_type = 'queen'
        func = {'rook': pysal.rook_from_shapefile,
                'queen': pysal.queen_from_shapefile}[cont_type]
        W = func(sfile, var)
        if order > 1:
            W_orig = W
            W = W.higher_order(order)
            if include_lower:
                for o in xrange(order - 1, 1, -1):
                    W = pysal.weights.w_union(W_orig.higher_order(o), W)
                W = pysal.weights.w_union(W, W_orig)
        W.meta = {'shape file': sfile,
                  'id variable': var,
                  'method': 'contiguity',
                  'method options': cont_type}
        self.SetW(W)

    def run_distance(self, sfile, var):
        """ Invoked by main run method. """
        if self.model.shapes.type == pysal.cg.Polygon:
            self.warn("The selected shapefile contains polygons and distance \
                      weights can only be computed on points. " +
                      "The centroids of the specified polygons will be used \
                      instead.")
        elif self.model.shapes.type != pysal.cg.Point:
            return self.warn("The selected shapefile does not contain points \
                             and contiguity weights can only be computed \
                             on points.")
        if self.model.distMethod == 0:
            radius = None
        elif self.model.distMethod == 1:  # 'Arc Distance (miles)'
            radius = pysal.cg.RADIUS_EARTH_MILES
        elif self.model.distMethod == 2:  # 'Arc Distance (kilometers)'
            radius = pysal.cg.RADIUS_EARTH_KM

        if self.ThresholdRadio.GetValue():
            try:
                cutoff = float(self.CutoffText.GetValue())
            except:
                return self.warn("The cut-off point is not valid.")
            print "Threshold on %s, ids=%r, cutoff=%f" % (sfile, var, cutoff)
            W = pysal.threshold_binaryW_from_shapefile(
                sfile, cutoff, idVariable=var, radius=radius)
            W.meta = {'shape file': sfile,
                      'id variable': var,
                      'method': 'distance',
                      'method options': ['Threshold', cutoff]}
            if radius:
                W.meta['Sphere Radius'] = radius
            self.SetW(W)
        elif self.KnnRadio.GetValue():
            k = int(self.NumNeighSpin.GetValue())
            print "Knn on %s, ids=%r, k=%d" % (sfile, var, k)
            W = pysal.knnW_from_shapefile(
                sfile, k=k, idVariable=var, radius=radius)
            W.meta = {'shape file': sfile,
                      'id variable': var,
                      'method': 'distance',
                      'method options': ['KNN', k]}
            if radius:
                W.meta['Sphere Radius'] = radius
            self.SetW(W)
        elif self.InverseRadio.GetValue():
            try:
                cutoff = float(self.CutoffText2.GetValue())
            except:
                return self.warn("The cut-off point is not valid.")
            power = int(self.PowerSpin.GetValue())
            print "Inverse on %s, ids=%r, cutoff=%f, power=%d" % \
                  (sfile, var, cutoff, power)
            try:
                W = pysal.threshold_continuousW_from_shapefile(
                    sfile, cutoff, alpha=- 1 * power,
                    idVariable=var, radius=radius)
                W.meta = {'shape file': sfile,
                          'id variable': var,
                          'method': 'distance',
                          'method options': ['Inverse', cutoff, power]}
                if radius:
                    W.meta['Sphere Radius'] = radius
                self.SetW(W)
            except Exception:
                et, e, tb = sys.exc_info()
                d = wx.MessageDialog(self, "\"%s\"\n"
                                     % str(e), "Error", wx.OK | wx.ICON_ERROR)
                d.ShowModal()

    def run_adaptive_kernel(self, sfile, var):
        """ Invoked by main run method. """
        if self.model.shapes.type == pysal.cg.Polygon:
            self.warn("The selected shapefile contains polygons and kernel \
                      weights can only be computed on points. " +
                      "The centroids of the specified polygons will be used\
                      instead.")
        elif self.model.shapes.type != pysal.cg.Point:
            return self.warn("The selected shapefile does not contain points\
                             and kernel weights can only be computed on \
                             points.")
        kern = ['uniform', 'triangular', 'quadratic', 'quartic', 'gaussian'][
            self.KFuncChoice.GetSelection()]
        k = int(self.KNumNeighSpin.GetValue())
        if self.model.distMethod == 0:
            radius = None
        elif self.model.distMethod == 1:  # 'Arc Distance (miles)'
            radius = pysal.cg.RADIUS_EARTH_MILES
        elif self.model.distMethod == 2:  # 'Arc Distance (kilometers)'
            radius = pysal.cg.RADIUS_EARTH_KM
        print "Kernel on %s, k=%d, ids=%r, kernel=%s" % (sfile, k, var, kern)
        W = pysal.adaptive_kernelW_from_shapefile(
            sfile, k=k, function=kern, idVariable=var, radius=radius)
        W = pysal.weights.insert_diagonal(W, wsp=False)
        W.meta = {'shape file': sfile,
                  'id variable': var,
                  'method': 'adaptive kernel',
                  'method options': [kern, k]}
        if radius:
            W.meta['Sphere Radius'] = radius
        self.SetW(W)

    def run(self, evtName=None, evt=None, value=None):
        if self.model.shapes is not None:
            sfile = self.model.inShps[self.model.inShp]
            method = self.weightsNotebook.GetPageText(
                self.weightsNotebook.GetSelection())
            var = None
            if self.model.idVar is not None:
                var = self.model.vars[self.model.idVar]
            if method == 'Contiguity':
                self.run_contiguity(sfile, var)
            elif method == 'Distance':
                self.run_distance(sfile, var)
            elif method == 'Adaptive Kernel':
                self.run_adaptive_kernel(sfile, var)
        if self.HasW():
            if self.requireSave:
                self.save()
            self.close()

    def close(self, ret_val=wx.ID_OK):
        if self.IsModal():
            self.EndModal(ret_val)
        else:
            self.Hide()

    def OnClose(self, evt):
        self.close(ret_val=wx.ID_CANCEL)

    def closeEvt(self, evtName=None, evt=None, value=None):
        self.close(ret_val=wx.ID_CANCEL)

    def distMeth(self, evtName=None, evt=None, value=None):
        if value is not None:
            self.DdistMethodChoice.SetSelection(value)
            self.KdistMethodChoice.SetSelection(value)
            self.distance_hints()
        else:
            self.model.distMethod = evt.EventObject.GetSelection()
