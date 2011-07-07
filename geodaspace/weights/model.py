#System
import os
#3rd Party
from scipy.spatial import KDTree
import numpy
import pysal
#Local
from geodaspace import AbstractModel
from geodaspace import DEBUG

class weightsModel(AbstractModel):
    def __init__(self):
        AbstractModel.__init__(self)
        self.reset()
        self.prop_reset()
    def reset(self):
        self._modelData['inShp'] = ''
        self._modelData['inShps'] = []
        self._modelData['idVar'] = None
    def prop_reset(self):
        self._propData = {}
    def __get_inShp(self):
        if DEBUG: print "getting inShp:",self._modelData.get('inShp','')
        return self._modelData.get('inShp','')
    def __set_inShp(self,value):
        if DEBUG: print "setting inShp:",value
        if not value == None:
            if type(value) == int: # change in inShp from current list
                self._modelData['inShp'] = value
            else: #elif type(value) == str: #add to list # or it could be unicode
                if value in self.inShps:
                    self._modelData['inShp'] = self.inShps.index(value)
                elif os.path.exists(value):
                    self.inShps = self.inShps+[value]
                    self._modelData['inShp'] = self.inShps.index(value)
            self.prop_reset()
            self.update('inShp')
    inShp = property(fget=__get_inShp,fset=__set_inShp)
    inShps = AbstractModel.abstractProp('inShps', list)
    idVar = AbstractModel.abstractProp('idVar', int)
    @property
    def vars(self):
        try:
            return self.data.header
        except:
            return []
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
            x,y,X,Y = self.shapes.bbox
            self._propData['bbox_diag'] = ((X-x)**2+(Y-y)**2)**(0.5)
            return self._propData['bbox_diag']
        return 0.0
    @property
    def knn1_dist(self):
        """ Returns the minimum distance required for each point to have at least one neighbor.
            This us the max dist returned by knn1 (k=2 because scipy includes each point as it's own neighbor)
        """
        if 'knn1_dist' in self._propData:
            return self._propData['knn1_dist']
        pts = self.points
        if pts != None:
            try:
                kd = KDTree(pts)
            except:
                leaf_size = len(set(map(tuple,pts.tolist())))
                kd = KDTree(pts,leaf_size)
            dists,ids = kd.query(pts,2)
            self._propData['knn1_dist'] = dists[:,1].max()
            return self._propData['knn1_dist']
        else:
            return 0.0
    @property
    def points(self):
        """ Attempts to return an array of points
            If the selected shapefile contains polygons the centroids will be returned.
        """
        if not self.shapes:
            return None
        shps = self.shapes
        if shps.type == pysal.cg.Polygon:
            return numpy.array([poly.centroid for poly in shps])
        elif shps.type == pysal.cg.Point:
            return numpy.array([pt for pt in shps])
        else:
            return None
    
if __name__ == '__main__':
    m = weightsModel()
    m.inShp = "/Users/charlie/Documents/data/stl_hom/stl_hom.shp"
