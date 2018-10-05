# GeoDaSpace
Software for Advanced Spatial Econometrics


GeoDaSpace current version 1.0 (32-bit)

Development environment:

Mac OSX 10.5.x (32-bit)

wxPython 2.8.x (which is 32-bit only version). On Mac OSX, only version < 10.6 (Lion) support 32-bit python by default. This cause a major problem to setup a dev environment on Mac OSX >= 10.6 (which is by default using 64-bit Python).

WxPython >=2.9.x has a impact on the GeoDaSpace UI, and the drag-n-drop function doesn't work anymore. See attached:

Bundling program: PyInstaller.


On Mac OSX 10.6+:

Install numpy 1.4.x
Install scipy 1.14.x
Install wxPython 2.9.x
