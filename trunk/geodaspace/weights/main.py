import wx
from view import xrcmainGUI
import control
from control import ENABLE_CONTIGUITY_WEIGHTS, ENABLE_DISTANCE_WEIGHTS, ENABLE_KERNEL_WEIGTHS

class mainGuiFrame(xrcmainGUI):
    def OnButton_ModelWeigths(self,evt):
        if not hasattr(self,"model_weights_diag"):
            self.model_weights_diag = control.weightsDialog(style = ENABLE_CONTIGUITY_WEIGHTS|ENABLE_DISTANCE_WEIGHTS)
        if self.model_weights_diag.ShowModal() == wx.ID_OK:
            my_w = self.model_weights_diag.GetW()
            print my_w
    def OnButton_KernelWeights(self,evt):
        if not hasattr(self,"kernel_weights_diag"):
            self.kernel_weights_diag = control.weightsDialog(style = ENABLE_KERNEL_WEIGTHS)
        if self.kernel_weights_diag.ShowModal() == wx.ID_OK:
            my_w = self.kernel_weights_diag.GetW()
            print my_w
    def OnButton_Quit(self,evt):
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
