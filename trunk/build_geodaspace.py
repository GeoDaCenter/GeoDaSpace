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
import time,datetime
import geodaspace.version
import sys,os
if __name__=='__main__':
    if len(sys.argv) == 1 or (len(sys.argv) == 2 and 'force' in sys.argv):
        print "Going to Build Now!"
        today = datetime.date(*time.localtime()[:3])
        if geodaspace.version.version_date != today:
            if 'force' in sys.argv:
                print "Today is:",today.isoformat()
                print "Version Release Date is:", geodaspace.version.version_date.isoformat()
                print "Please update the version information before building!"
                print "But if you are going to force me to build..."
                idx = sys.argv.index('force')
                sys.argv.pop(idx)
                
            else:
                print 
                print "Sorry, can not build now!"
                print "Today is:",today.isoformat()
                print "Version Release Date is:", geodaspace.version.version_date.isoformat()
                print "Please update the version information before building!"
                print "You can force a build by running:"
                print "\t python build_geodaspace.py force"
                sys.exit(1)


        if sys.platform == 'darwin':
            sys.argv.extend(['py2app', '--iconfile', 'geodaspace/icons/geodaspace.icns'])
        elif sys.platform == 'win32':
            sys.argv.extend(['py2exe', '--bundle=1'])

from distutils.core import setup
pkgs = []
if sys.platform == 'darwin':
    import py2app
    setup( app=['geodaspace/GeoDaSpace.py'], 
           packages=pkgs,
           name='GeoDaSpace',
           version=geodaspace.version.version,
         )
    os.rename('dist/GeoDaSpace.app','dist/GeoDaSpace OSX %s.app'%geodaspace.version.version)
elif sys.platform == 'win32':
    import py2exe
    from glob import glob
    sys.path.append(r".\Microsoft.VC90.CRT")
    setup( zipfile=None,
           windows=[ {"script": "geodaspace/GeoDaSpace.py", 
                      "icon_resources": [(1, "geodaspace/icons/geodaspace.ico")]}],
                                
           data_files=[('Microsoft.VC90.CRT',glob('Microsoft.VC90.CRT/*.*'))],
           # remove the dll_excludes entry if you want to support Win95/98
           options = {'py2exe': { 
                        'includes': ['scipy.io.matlab.streams'],
                        'dll_excludes': ['w9xpopen.exe'],
                        'excludes' : ['Tkconstants', 'Tkinter', 'tcl'],                        
                        } },
           packages= pkgs,
           name='GeoDaSpace',
           version=geodaspace.version.version,
         )
    os.rename('dist/GeoDaSpace.exe','dist/GeoDaSpace Windows %s.exe'%geodaspace.version.version)
