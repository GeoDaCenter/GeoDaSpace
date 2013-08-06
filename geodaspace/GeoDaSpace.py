import wx
from geodaspace.regression import variableTools
from geodaspace.regression import M_regression
from geodaspace.regression import V_regression
import sys
import multiprocessing

multiprocessing.freeze_support()

if sys.platform == 'darwin':
    DOCK_SIZE = 40
else:
    DOCK_SIZE = 80


class guiApp(wx.App):
    """ Entry Point for GMM Regression GUI """
    results = []

    def __init__(self, redirect=False, filename=None):
        wx.App.__init__(self, redirect, filename)

    def OnInit(self):
        # Load the Frame
        x, y, X, Y = wx.ClientDisplayRect()
        maxWidth = X - x
        maxHeight = Y - y
        self.regFrame = V_regression.guiRegView(results=self.results)
        width, height = self.regFrame.GetSize()
        if height > maxHeight:
            height = maxHeight - DOCK_SIZE
            width += 15  # Scroll bar width
        if width > maxWidth:
            width = maxWidth
        if (width, height) != self.regFrame.GetSize():
            self.regFrame.SetClientRect((0, 0, width, height))
            self.regFrame.SetPosition((x, y))

        self.regFrame.Show()
        self.SetTopWindow(self.regFrame)
        self.regFrame.scroll.SetVirtualSize((570, 647))
        self.regFrame.scroll.SetScrollRate(1, 1)
        return True

def main():
    app = guiApp()
    app.MainLoop()

if __name__ == "__main__":
    """ usage: python -i main.py """
    #app = guiApp()
    #app.MainLoop()
    main()
