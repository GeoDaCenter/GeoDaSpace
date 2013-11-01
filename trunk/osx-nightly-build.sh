#!/bin/bash
PYTHONPATH=~/Desktop/pysal:~/Desktop/spreg/trunk
export PYTHONPATH
PATH=/usr/local/git/bin:/Library/Frameworks/Python.framework/Versions/2.7/bin:$PATH
export PATH


cd ~/Desktop/pysal
git fetch

cd ~/Desktop/spreg/trunk
svn update

python  build_geodaspace.py 
cd ~/Desktop/spreg/trunk/dist/
tar -czvf GeoDaSpace_OSX_Nightly.tar.gz GeoDaSpace\ OSX\ 0.8.6.app/
cp GeoDaSpace_OSX_Nightly.tar.gz /Volumes/GeoDa/Projects/GeoDaSpace/Nightly/


