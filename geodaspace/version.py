import datetime
def get_long_version():
    s = ""
    if version_type == 'alpha':
        s+="(alpha) "
    elif version_type == 'beta':
        s+="(beta) "
    s+="revision "
    s+=version
    s+="; "
    s+=version_date.strftime('%B %d, %Y')
    return s

#Version History
#version = "0.7.0" #spreg r682, pysal r1069
#version_type = "alpha"
#version_date = datetime.date(2011,12,14)
#version = "0.7.1" #spreg r683, pysal r1070
#version_type = "alpha"
#version_date = datetime.date(2011,12,15)
#Current Version
#version = "0.7.2" #spreg r683, pysal r1070
#version_type = "alpha"
#version_date = datetime.date(2011,12,23)
#version = "0.7.3" #spreg r688, pysal r1070
#version_type = "alpha"
#version_date = datetime.date(2012, 1, 9)

version = "0.7.3.1" #spreg r707, pysal r1157
version_type = "alpha"
version_date = datetime.date(2012, 2, 7)
