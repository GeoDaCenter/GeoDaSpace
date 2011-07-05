
import math

def comp_arc_dist(lat1,long1,lat2,long2,unit='miles'):
    # check incoming data
    for i in [lat1,lat2]:
        if i > 90:
            raise Exception, "Latitude values must be in the range [-90,90]"
        if i < -90:
            raise Exception, "Latitude values must be in the range [-90,90]"
    for i in [long1,long2]:
        if i > 180:
            raise Exception, "Longitude values must be in the range [-180,180]"
        if i < -180:
            raise Exception, "Longitude values must be in the range [-180,180]"
    # set constants
    pi = math.pi
    if unit == 'miles':
        erad = 3959.0
    elif unit == 'kilometers': 
        erad = 6371.0
    else: 
        raise Exception, "Either miles or kilometers must be selected for units."
    # compute the arc distance
    rlat1 = (90 - lat1)*math.pi/180.0
    rlat2 = (90 - lat2)*math.pi/180.0
    rlong = (long2-long1)*math.pi/180.0
    drad = math.cos(rlong)*math.sin(rlat1)*math.sin(rlat2) + math.cos(rlat1)*math.cos(rlat2)
    dist = math.acos(drad)*erad 
    return dist
