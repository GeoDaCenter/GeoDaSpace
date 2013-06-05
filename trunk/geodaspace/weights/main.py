import wx
from view import xrcmainGUI
import control
from control import ENABLE_CONTIGUITY_WEIGHTS, ENABLE_DISTANCE_WEIGHTS, ENABLE_KERNEL_WEIGTHS
from control import WEIGHT_TYPES_FILTER, WEIGHT_FILTER_TO_HANDLER
from model import GeoDaSpace_W_Obj
import pysal


class mainGuiFrame(xrcmainGUI):
    def OnButton_ModelWeigths(self, evt):
        if not hasattr(self, "model_weights_diag"):
            self.model_weights_diag = control.weightsDialog(
                style=ENABLE_CONTIGUITY_WEIGHTS | ENABLE_DISTANCE_WEIGHTS)
        if self.model_weights_diag.ShowModal() == wx.ID_OK:
            my_w = self.model_weights_diag.GetW()
            print my_w

    def OnButton_KernelWeights(self, evt):
        if not hasattr(self, "kernel_weights_diag"):
            self.kernel_weights_diag = control.weightsDialog(
                style=ENABLE_KERNEL_WEIGTHS)
        if self.kernel_weights_diag.ShowModal() == wx.ID_OK:
            my_w = self.kernel_weights_diag.GetW()
            print my_w

    def OnButton_WeightsProps(self, evt):
        fileDialog = wx.FileDialog(
            self, message="Choose File", wildcard=WEIGHT_TYPES_FILTER,
            style=wx.FD_OPEN | wx.FD_MULTIPLE)
        if fileDialog.ShowModal() == wx.ID_OK:
            handler = WEIGHT_FILTER_TO_HANDLER[fileDialog.GetFilterIndex()]
            w_objs = []
            for path in fileDialog.GetPaths():
                print path
                W = GeoDaSpace_W_Obj.from_path(path, handler)
                w_objs.append(W)
            print w_objs
            wpropDlg = control.weightsPropertiesDialog(self)
            wpropDlg.ShowModal(w_objs)

    def OnButton_Quit(self, evt):
        self.Destroy()


class SimpleStandaloneApp(wx.App):
    def OnInit(self):
        self.frame = mainGuiFrame(None)
        self.frame.Show()
        self.SetTopWindow(self.frame)
        return True


def run():
    app = SimpleStandaloneApp(redirect=False)
    app.MainLoop()

if __name__ == '__main__':
    run()
