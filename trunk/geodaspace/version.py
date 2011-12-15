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

#Current Version
version = "0.7.1" #spreg r683, pysal r1070
version_type = "alpha"
version_date = datetime.date(2011,12,15)
