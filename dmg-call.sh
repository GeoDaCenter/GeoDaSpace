#! /bin/bash
# This is the command line call to create the fancy dmg for packaging geodaspace
# or other software for mac os x redistribution

# original
#test -f test2.dmg && rm test2.dmg
#./create-dmg --window-size 500 300 --background ~/Projects/eclipse-osx-repackager/build/background.gif --icon-size 96 --volname "Hyper Foo" --app-drop-link 380 205 --icon "Eclipse OS X Repackager" 110 205 test2.dmg /Users/andreyvit/Projects/eclipse-osx-repackager/temp/Eclipse\ OS\ X\ Repackager\ r10/

test -f GeoDaSpace.dmg && rm GeoDaSpace.dmg 
./create-dmg --window-size 500 300 --background background.png  --volname "GeoDaSpace" --app-drop-link 380 205  GeoDaSpace.dmg /Users/stephens/Dropbox/work/Projects/spreg/trunk/dist
