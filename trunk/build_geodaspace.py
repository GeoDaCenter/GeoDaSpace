#!/usr/bin/env python
"""
setup.py -- script for building binary package of GeoDaSpace GUI

Usage:
    Mac:
    % python setup.py py2app
    Windows:
    c:\> python setup.py py2exe --bundle=1

        Notes for Windows: Do not try to build on a shared or otherwise network drive.
                           Make sure the the build location is on a local drive.
"""
import sys,os
if __name__=='__main__':
    if len(sys.argv) == 1:
        print "Going to Build Now!"
        if sys.platform == 'darwin':
            sys.argv.extend(['py2app', '--iconfile', 'geodaspace/icons/geodaspace.icns'])
        elif sys.platform == 'win32':
            sys.argv.extend(['py2exe', '--bundle=1'])

from distutils.core import setup
pkgs = []
if sys.platform == 'darwin':
    import py2app
    setup( app=['geodaspace/GeoDaSpace.py'], 
           packages=pkgs
         )
elif sys.platform == 'win32':
    import py2exe
    from glob import glob
    sys.path.append(r".\Microsoft.VC90.CRT")
    setup( zipfile=None,
           windows=[ {"script": "geodaspace/GeoDaSpace.py", 
                      "icon_resources": [(1, "geodaspace/icons/geodaspace.ico")]}],
           data_files=[('Microsoft.VC90.CRT',glob('Microsoft.VC90.CRT/*.*'))],
           options = {'py2exe': { 
                        'includes': ['scipy.io.matlab.streams']
                        } },
           packages= pkgs
         )

