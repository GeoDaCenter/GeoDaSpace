import wx
from control import preferencesFrame

class SimpleStandaloneApp(wx.App):
    def OnInit(self):
        self.frame = preferencesFrame()
        self.frame.Show()
        self.SetTopWindow(self.frame)
        return True

def run():
    app = SimpleStandaloneApp(redirect=False)
    app.MainLoop()

if __name__ == '__main__':
    run()
