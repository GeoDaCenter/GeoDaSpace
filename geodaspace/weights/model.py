#System
import os
#3rd Party
from pysal.common import KDTree
import numpy
import pysal
#Local
from geodaspace import AbstractModel
from geodaspace import DEBUG

DISTANCE_METRICS = ['Euclidean Distance','Arc Distance (miles)', 'Arc Distance (kilometers)']

class weightsModel(AbstractModel):
    def __init__(self):
        AbstractModel.__init__(self)
        self.reset()
        self.prop_reset()
    def reset(self):
        self._modelData['inShp'] = ''
        self._modelData['inShps'] = []
        self._modelData['idVar'] = None
        self._modelData['distMethod'] = 0
    def prop_reset(self):
        self._propData = {}
    def __get_inShp(self):
        if DEBUG: print "getting inShp:",self._modelData.get('inShp','')
        return self._modelData.get('inShp','')
    def __set_inShp(self,value):
        if DEBUG: print "setting inShp:",value
        if not value == None:
            if type(value) == int: # change in inShp from current list
                if self._modelData['inShp'] == value:
                    return
                self._modelData['inShp'] = value
            else: #elif type(value) == str: #add to list # or it could be unicode
                if value in self.inShps:
                    self._modelData['inShp'] = self.inShps.index(value)
                elif os.path.exists(value):
                    self.inShps = self.inShps+[value]
                    self._modelData['inShp'] = self.inShps.index(value)
            self.prop_reset()
            self._modelData['idVar'] = None
            self.update('inShp')
    inShp = property(fget=__get_inShp,fset=__set_inShp)
    inShps = AbstractModel.abstractProp('inShps', list)
    idVar = AbstractModel.abstractProp('idVar', int)
    def __get_distMethod(self):
        return self._modelData['distMethod']
    def __set_distMethod(self,val):
        if type(val) == int:
            self._modelData['distMethod'] = val
            self.prop_reset()
            self.update('distMethod')
    distMethod = property(fget=__get_distMethod, fset=__set_distMethod)
    @property
    def vars(self):
        try:
            return self.data.header
        except:
            return []
    @property
    def data_path(self):
        try:
            return self.inShps[self.inShp][:-4]+'.dbf'
        except:
            return None
    @property
    def data(self):
        try:
            return pysal.open(self.inShps[self.inShp][:-4]+'.dbf','r')
        except:
            return None
    @property
    def shapes(self):
        try:
            return pysal.open(self.inShps[self.inShp],'r')
        except:
            return None
    @property
    def bbox_diag(self):
        """ Returns the length of the diagonal of the bounding box the selected shapefile or 0.0 """
        if 'bbox_diag' in self._propData:
            return self._propData['bbox_diag']
        if self.shapes:
            if self.distMethod == 0: #'Euclidean Distance'
                x,y,X,Y = self.shapes.bbox
                self._propData['bbox_diag'] = ((X-x)**2+(Y-y)**2)**(0.5)
                return self._propData['bbox_diag']
            elif self.distMethod == 1: #'Arc Distance (miles)'
                x,y,X,Y = self.shapes.bbox
                self._propData['bbox_diag'] = pysal.cg.arcdist((y,x),(Y,X),radius=pysal.cg.RADIUS_EARTH_MILES)
                return self._propData['bbox_diag']
            elif self.distMethod == 2: #'Arc Distance (kilometers)'
                x,y,X,Y = self.shapes.bbox
                self._propData['bbox_diag'] = pysal.cg.arcdist((y,x),(Y,X),radius=pysal.cg.RADIUS_EARTH_KM)
                return self._propData['bbox_diag']
        return 0.0
    @property
    def knn1_dist(self):
        """ Returns the minimum distance required for each point to have at least one neighbor.
            This us the max dist returned by knn1 (k=2 because scipy includes each point as it's own neighbor)
        """
        if 'knn1_dist' in self._propData:
            return self._propData['knn1_dist']
        if self.kdtree:
            max_k1d = pysal.weights.util.min_threshold_distance(self.kdtree)
            self._propData['knn1_dist'] = max_k1d
            return self._propData['knn1_dist']
        else:
            return 0.0
    @property
    def kdtree(self):
        if 'kd' in self._propData:
            return self._propData['kd']
        else:
            pts = self.points
            if pts != None:
                if self.distMethod == 0: # not Euclidean Distance
                    kd = KDTree(pts)
                elif self.distMethod == 1: #'Arc Distance (miles)'
                    kd = KDTree(pts, distance_metric="Arc",radius = pysal.cg.RADIUS_EARTH_MILES)
                elif self.distMethod == 2: #'Arc Distance (kilometers)'
                    kd = KDTree(pts, distance_metric="Arc",radius = pysal.cg.RADIUS_EARTH_KM)
                self._propData['kd'] = kd
                return kd
        return None
    @property
    def points(self):
        """ Attempts to return an array of points
            If the selected shapefile contains polygons the centroids will be returned.
        """
        if not self.shapes:
            return None
        shps = self.shapes
        if shps.type == pysal.cg.Polygon:
            pts = [poly.centroid for poly in shps]
        elif shps.type == pysal.cg.Point:
            pts = [pt for pt in shps]
        else:
            return None
        #if self.distMethod != 0: # not Euclidean Distance
        #    pts = map(pysal.cg.toXYZ,pts)
        return numpy.array(pts)
    
if __name__ == '__main__':
    m = weightsModel()
    m.inShp = "/Users/charlie/Documents/data/stl_hom/stl_hom.shp"
