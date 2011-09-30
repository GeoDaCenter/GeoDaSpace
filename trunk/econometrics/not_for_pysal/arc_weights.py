"""
Arc weights: compute weights for lat/lng pairs assuming a spherical earth.
Author: Charles R Schmidt <charlies.r.schmidt@asu.edu>
"""
import math
import random
import numpy
import scipy.spatial
from math import pi,cos,sin,asin

from arcdist import comp_arc_dist

RADIUS_EARTH_MILES = 3959.0
RADIUS_EARTH_KM = 6371.0

def comp_xyz_dist(pt0,pt1):
    x,y,z = pt0
    X,Y,Z = pt1
    return ((x-X)**2 + (y-Y)**2 + (z-Z)**2) ** (0.5)

def random_ll():
    long = (random.random()*360) - 180
    lat = (random.random()*180) - 90
    return long,lat

def arcdist_to_linear(arc_dist, unit = 'miles'):
    """
    Convert an arc distance (spherical earth) to a linear distance (R3)

    Doc Test
    >>> lat0,lng0 = 0,0
    >>> lat1,lng1 = 0,180
    >>> d = comp_arc_dist(lat0,lng0,lat1,lng1)
    >>> print '%0.2f'%d
    12437.57
    >>> comp_xyz_dist(toXYZ((lng0,lat0)), toXYZ((lng1,lat1)))
    2.0
    >>> arcdist_to_linear(d)
    2.0
    """
    if unit == 'miles':
        r = RADIUS_EARTH_MILES
    elif unit == 'kilometers': 
        r = RADIUS_EARTH_KM
    else: 
        raise Exception, "Either miles or kilometers must be selected for units."
    c = 2*math.pi*r
    d = (2-(2*math.cos(math.radians((arc_dist*360.0)/c)))) ** (0.5)
    return d
    


def toXYZ(pt):
    """ ASSUMPTION: pt = (lng,lat)
        REASON: pi = 180 degress,
                theta+(pi/2)....
                theta = 90 degrees,
                180 =  90+180/2"""
    phi,theta = map(math.radians,pt)
    phi,theta = phi+pi,theta+(pi/2)
    x = 1*sin(theta)*cos(phi)
    y = 1*sin(theta)*sin(phi)
    z = 1*cos(theta)
    return x,y,z

def toLngLat(xyz):
    x,y,z = xyz
    if z == -1 or z == 1:
        phi = 0
    else:
        phi = math.atan2(y,x)
        if phi > 0:
            phi = phi-math.pi
        elif phi < 0:
            phi = phi+math.pi
    theta = math.acos(z)-(math.pi/2)
    return phi,theta

def brute_knn(pts,k,mode='arc'):
    """
    valid modes are ['arc','xrz']
    """
    n = len(pts)
    full = numpy.zeros((n,n))
    for i in xrange(n):
        for j in xrange(i+1,n):
            if mode == 'arc':
                lng0,lat0= pts[i]
                lng1,lat1= pts[j]
                dist = comp_arc_dist(lat0,lng0,lat1,lng1,unit='kilometers')
            elif mode == 'xyz':
                dist = comp_xyz_dist(pts[i],pts[j])
            full[i,j] = dist
            full[j,i] = dist
    w = {}
    for i in xrange(n):
        w[i] = full[i].argsort()[1:k+1].tolist()
    return w
def fast_knn(pts,k):
    pts = numpy.array(pts)
    kd = scipy.spatial.KDTree(pts)
    d,w = kd.query(pts,k+1)
    w = w[:,1:]
    wd = {}
    for i in xrange(len(pts)):
        wd[i] = w[i].tolist()
    return wd
def fast_threshold(pts,dist,unit='miles'):
    d = arcdist_to_linear(dist,unit)
    kd = scipy.spatial.KDTree(pts)
    r = kd.query_ball_tree(kd,d)
    wd = {}
    for i in xrange(len(pts)):
        l = r[i]
        l.remove(i)
        wd[i] = l
    return wd



if __name__=='__main__':
    for i in range(1):
        n = 99
        # generate random surface points.
        pts = [random_ll() for i in xrange(n)]
        # convert to unit sphere points.
        pts2 = map(toXYZ, pts)
        
        w = brute_knn(pts,4,'arc')
        w2 = brute_knn(pts2,4,'xyz')
        w3 = fast_knn(pts2,4)
        assert w == w2 == w3
    import doctest
    doctest.testmod()
