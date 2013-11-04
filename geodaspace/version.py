import datetime
import subprocess

version_date = datetime.date.today()
# Current Release
version = "0.8.6"  # spreg r1018, pysal 1.7.0dev

# toggle these 2 below for official alpha releases
#version_type = "alpha"
version_type = 'nightly'

def run_cmd(cmd):
    p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
    output = p.communicate()[0]
    return output

spreg_revision = run_cmd("tail -1 /Users/gspace/Desktop/spreg/trunk/rev.txt")
spreg_revision = spreg_revision.strip('\n')
#spreg_revision = 'Revision: 1029'

def get_long_version():
    s = ""
    if version_type == 'nightly':
        s += "(nightly) "
    if version_type == 'alpha':
        s += "(alpha) "
    elif version_type == 'beta':
        s += "(beta) "
    s += "Release "
    s += version
    if version_type == 'nightly':
        s += " | "
        s += "spReg "
        s += spreg_revision
    s += " | "
    s += version_date.strftime('%B %d, %Y')
    return s

# Version History
# version = "0.7.0" #spreg r682, pysal r1069
# version_date = datetime.date(2011,12,14)
# version = "0.7.1" #spreg r683, pysal r1070
# version_date = datetime.date(2011,12,15)
# version = "0.7.2" #spreg r683, pysal r1070
# version_date = datetime.date(2011,12,23)
# version = "0.7.3" #spreg r688, pysal r1070
# version_date = datetime.date(2012, 1, 9)
# version = "0.7.3.1" #spreg r707, pysal r1157
# version_date = datetime.date(2012, 2, 7)
# version = "0.7.4" #spreg r770, pysal r1334
# version_date = datetime.date(2012, 8, 13)
# version = "0.7.5" #spreg r786, pysal r1353
# version_date = datetime.date(2012, 9, 6)
# version = "0.7.6"  #spreg r798, pysal r1353
# version_date = datetime.date(2012, 9, 7)
# version = "0.7.7"  #spreg r854, pysal r1403
# version_date = datetime.date(2012, 12, 6)
# version = "0.7.8"  #spreg r886, pysal r1450
# version_date = datetime.date(2013, 2, 22)
# version = "0.7.9"  #spreg r898, pysal 1.6.0dev
# version_date = datetime.date(2013, 4, 22)
# version = "0.8.0"  #spreg r904, pysal 1.6.0dev
# version_date = datetime.date(2013, 4, 26)
# version = "0.8.1"  # spreg r918, pysal 1.6.0dev
# version_date = datetime.date(2013, 5, 29)
# version = "0.8.2"  # spreg r970, pysal 1.6.0dev
# version_date = datetime.date(2013, 7, 3)
# version = "0.8.3"  # spreg r978, pysal 1.6.0dev
# version_date = datetime.date(2013, 7, 29)
# version = "0.8.4"  # spreg r984, pysal 1.7.0dev
# version_date = datetime.date(2013, 8, 6)
# version = "0.8.5"  # spreg r1014, pysal 1.7.0dev
# version_date = datetime.date(2013, 10, 2)

