import datetime

# version_date = datetime.date.today()  # doing this causes the date string in the gui to update every day it is run!
# Current Release. Note: when you bump, change srcfolder in osx-nightly-build script
version = "1.0"  # spreg r1083, pysal 1.7.0dev

# toggle these 2 below for official alpha releases
version_type = "alpha"
#version_type = 'nightly'

def get_long_version():
    s = ""
    if version_type == 'nightly':
        s += "(nightly) "
        s += " | "
        s += "spReg revision "
        s += rev
        s += " | "
        version_date = datetime.date.today()
        s += version_date.strftime('%B %d, %Y')
        s += " | "
    if version_type == 'alpha':
        s += "(alpha) "
        s += " | "
        version_date = datetime.date(2014, 6, 1)
        s += version_date.strftime('%B %d, %Y')
        s += " | "
    elif version_type == 'beta':
        s += "(beta) "
    s += "Release "
    s += version
    return s

# Version History

rev = '1112'
