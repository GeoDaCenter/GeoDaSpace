#!/bin/bash
PYTHONPATH=~/Desktop/pysal:~/Desktop/spreg/trunk
export PYTHONPATH
PATH=/usr/local/git/bin:/Library/Frameworks/Python.framework/Versions/2.7/bin:$PATH
export PATH

rm /Volumes/GeoDa/Projects/GeoDaSpace/Nightly/GeoDaSpace_OSX_Nightly.tar.gz

cd ~/Desktop/pysal
git fetch

cd ~/Desktop/spreg/trunk
/usr/bin/make clean
svn update
#svnversion >> ~/Desktop/spreg/trunk/rev.txt
svn info | grep 'Revision' >> ~/Desktop/spreg/trunk/rev.txt

python  build_geodaspace.py 
cd ~/Desktop/spreg/trunk/dist/
tar -czvf GeoDaSpace_OSX_Nightly.tar.gz GeoDaSpace\ OSX\ 0.8.6.app/
cp GeoDaSpace_OSX_Nightly.tar.gz /Volumes/GeoDa/Projects/GeoDaSpace/Nightly/


