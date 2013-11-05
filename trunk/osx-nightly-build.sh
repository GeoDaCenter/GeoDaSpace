#!/bin/bash

# make local libs available to python
PYTHONPATH=~/Desktop/pysal:~/Desktop/spreg/trunk
export PYTHONPATH

# set path to Git and preferred Python
PATH=/usr/local/git/bin:/Library/Frameworks/Python.framework/Versions/2.7/bin:$PATH
export PATH

# delete last nightly build
rm /Volumes/GeoDa/Projects/GeoDaSpace/Nightly/*

# fetch changes in pysal
cd ~/Desktop/pysal
git fetch

# clean up old build data
cd ~/Desktop/spreg/trunk
/usr/bin/make clean

# retrieve changes in spreg 
svn update

# paste svn version number into version script
echo "rev = '$(svn info | grep Revision: | cut -c11-)'" >> ~/Desktop/spreg/trunk/geodaspace/version.py

# build
python  build_geodaspace.py 
cd ~/Desktop/spreg/trunk/dist/

# create the disk image
hdiutil create -fs HFS+ -srcfolder GeoDaSpace\ OSX\ 0.8.6.app/ GeoDaSpace_OSX_Nightly.dmg

# copy to file server
cp GeoDaSpace_OSX_Nightly.dmg /Volumes/GeoDa/Projects/GeoDaSpace/Nightly/

