import os.path
#import wx
from geodaspace.abstractmodel import AbstractModel
import pysal


DTYPE = 'listvars'
FORMATHEADER = 0


class M_CreateSpatialLag(AbstractModel):
    DATA_KEYS_ORDER = ['vars', 'wtFiles', 'wtFile', 'dataFile']

    def __init__(self):
        AbstractModel.__init__(self)
        self.dispatch = {
            'wtFile': self.__wtFile, 'dataFile': self.__dataFile}  # dispatch
        self.reset()

    def reset(self):
        self.data = {}
        self.data['wtFiles'] = ['']
        self.data['wtFile'] = None
        self.data['vars'] = []
        self.newVars = []
        self.newVarNames = []

    def addVar(self, var):
        self.newVars.append(var)
        var.set('vars', self.get('vars'))
        var.addListener(self.varChanged)

    def varChanged(self, varMdl):
        newVarNames = [var.get('newVarName') for var in self.newVars]
        if newVarNames is not self.newVarNames:
            for var in self.newVars:
                if var.newVars == newVarNames:
                    pass
                else:
                    var.newVars = newVarNames
                    var.update()  # method from AbstractModel
            self.newVarNames = newVarNames

    def get(self, key=None):
        if key:
            if key in self.data:
                return self.data[key]
            else:
                print "Model Error: %s, not in model" % key
                return None
        else:
            for key in self.data:
                if not key in self.DATA_KEYS_ORDER:
                    print "Model Warning: %s, not ORDERED" % key
                    self.DATA_KEYS_ORDER.append(key)
            for key in self.DATA_KEYS_ORDER:
                if not key in self.data:
                    print "Model Warning: %s, not in data" % key
                    self.data[key] = None
            return self.data

    def set(self, key=None, value=None, passive=False):
        print "Setting", key, value
        if key in self.dispatch:
            self.dispatch[key](value)
        else:
            self.data[key] = value
        if not passive:
            self.update()  # method from AbstractModel

    def verify(self):
        """ Verify the model before running """
        if self.data['dataFile']:
            return True

    def run(self, path):
        if self.verify():
            print "running"
            # print self.data['wtFiles'][self.data['wtFile']]
            newVars = [var.run() for var in self.newVars]
            names = [v[0] for v in newVars]
            vars = [v[1] for v in newVars]
            db = self.db()
            xid = [db.header.index(i) for i in vars]
            X = [db[:, i] for i in xid]
            W = self.loadWeights()

            new_header = db.header + names
            new_spec = db.field_spec + [('N', 20, 10) for n in names]
            data = db.read()
            db.close()
            newdb = pysal.open(path, 'w')
            newdb.header = new_header
            newdb.field_spec = new_spec

            lag = [pysal.lag_spatial(W, y) for y in X]
            lag = zip(*lag)  # transpose
            lag = map(list, lag)
            for i, row in enumerate(data):
                newdb.write(row + lag[i])
            newdb.close()

    def db(self):
        return pysal.open(self.data['dataFile'], 'r')

    def loadWeights(self):
        wtFile = self.data['wtFiles'][self.data['wtFile']]
        if issubclass(type(wtFile), basestring):
            W = pysal.open(wtFile, 'r').read()
            W.transform = 'r'  # see issue #138
            return W
        else:
            W = wtFile.w
            if W.transform != 'r':
                W.transform = 'r'
            return W

    def __dataFile(self, value=None):
        if value is not None:
            if os.path.exists(value) and not os.path.isdir(value):
                self.data['dataFile'] = value
                db = self.db()
                for v in self.newVars:
                    v.set('vars', db.header)
                self.set('vars', db.header)
            else:
                self.data['dataFile'] = False

    def __wtFile(self, value=None):
        if value is not None:
            if type(value) == int:  # change the current wtFile from the GUI
                self.data['wtFile'] = value
            else:  # add str to list
                if not value in self.data['wtFiles']:
                    self.data['wtFiles'].append(value)
                self.data['wtFile'] = self.data['wtFiles'].index(value)


class M_spLagVariable(AbstractModel):
    """ Model for an XRC panel that contains, [textCtrl] = W*[dropDown] """
    DATA_KEYS_ORDER = ['vars', 'var', 'newVarName', 'caution']

    def __init__(self):
        AbstractModel.__init__(self)
        # dispatch
        self.dispatch = {'vars': self.__vars}
        self.reset()

    def reset(self):
        self.data = {}
        self.data['vars'] = ['']
        self.data['var'] = -1
        self.data['newVarName'] = ''
        self.data['caution'] = False
        self.newVars = []
        self.varsChanged = True

    def run(self):
        return self.data['newVarName'], self.data['vars'][self.data['var']]

    def get(self, key=None):
        if key:
            if key in self.data:
                return self.data[key]
            else:
                print "Model Error: %s, not in model" % key
                return None
        else:
            for key in self.data:
                if not key in self.DATA_KEYS_ORDER:
                    print "Model Warning: %s, not ORDERED" % key
                    self.DATA_KEYS_ORDER.append(key)
            for key in self.DATA_KEYS_ORDER:
                if not key in self.data:
                    print "Model Warning: %s, not in data" % key
                    self.data[key] = None
            return self.data

    def set(self, key=None, value=None, passive=False):
        print "M_spLagVar Setting", key, value
        if key in self.dispatch:
            self.dispatch[key](value)
        else:
            self.data[key] = value
        if not passive:
            self.update()  # method from AbstractModel

    def __vars(self, value=None):
        if value is not None:
            self.data['vars'] = value
            self.varsChanged = True
