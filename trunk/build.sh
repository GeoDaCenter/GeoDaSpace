#!/bin/bash
PYTHONPATH=/Users/gspace/Desktop/pysal:/Users/gspace/Desktop/spreg/trunk
export PYTHONPATH
#echo $PYTHONPATH
cd /Users/gspace/Desktop/pysal; /usr/local/git/bin/git fetch

cd /Users/gspace/Desktop/spreg/trunk
svn update

/Library/Frameworks/Python.framework/Versions/2.7/bin/python  build_geodaspace.py 
cd /Users/gspace/Desktop/spreg/trunk/dist/
tar -czvf GeoDaSpace_OSX_Nightly.tar.gz GeoDaSpace\ OSX\ 0.8.6.app/
cp GeoDaSpace_OSX_Nightly.tar.gz /Volumes/GeoDa/Projects/GeoDaSpace/Nightly/


Revision: 1025
Revision: 1025
Revision: 1025
