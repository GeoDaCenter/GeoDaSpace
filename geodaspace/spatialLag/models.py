import os.path
import wx
from geodaspace.abstractmodel import AbstractModel
from Lpysal import weights
from Lpysal import csvreader
from Lpysal import dbfreader


DTYPE = 'listvars'
FORMATHEADER = 0

class M_CreateSpatialLag(AbstractModel):
    DATA_KEYS_ORDER = ['vars','wtFiles','wtFile','dataFile']
    def __init__(self):
        AbstractModel.__init__(self)
        #dispatch
        self.dispatch = {'wtFile':self.__wtFile,'dataFile':self.__dataFile}
        self.reset()
    def reset(self):
        self.data = {}
        self.data['wtFiles'] = ['']
        self.data['wtFile'] = None
        self.data['vars'] = []
        self.newVars = []
        self.newVarNames = []
    def addVar(self,var):
        self.newVars.append(var)
        var.set('vars',self.get('vars'))
        var.addListener(self.varChanged)
    def varChanged(self,varMdl):
        newVarNames = [var.get('newVarName') for var in self.newVars]
        if newVarNames is not self.newVarNames:
            for var in self.newVars:
                if var.newVars == newVarNames:
                    pass
                else:
                    var.newVars=newVarNames
                    var.update()
            self.newVarNames = newVarNames
    def get(self,key=None):
        if key:
            if key in self.data:
                return self.data[key]
            else:
                print "Model Error: %s, not in model"%key
                return None
        else:
            for key in self.data:
                if not key in self.DATA_KEYS_ORDER:
                    print "Model Warning: %s, not ORDERED"%key
                    self.DATA_KEYS_ORDER.append(key)
            for key in self.DATA_KEYS_ORDER:
                if not key in self.data:
                    print "Model Warning: %s, not in data"%key
                    self.data[key] = None
            return self.data
    def set(self,key=None,value=None,passive=False):
        print "Setting",key,value
        if key in self.dispatch:
            self.dispatch[key](value)
        else:
            self.data[key] = value
        if not passive:
            self.update()
    def verify(self):
        """ Verify the model before running """
        if self.data['dataFile']:
            return True
    def run(self,path):
        if self.verify():
            print "running"
            #print self.data['wtFiles'][self.data['wtFile']]
            newVars = [var.run() for var in self.newVars]
            names = [v[0] for v in newVars]
            vars = [v[1] for v in newVars]
            db = self.db()
            xid = [ db.varnames.index(i) for i in vars ]
            X = [db.records[i] for i in xid]
            W = self.loadWeights(self.db())
            newdb = self.db()
            newdb.makedictvars()
            for field,values in zip(names, W.wsplagl(X)):
                if field: #else: field = '' and should be ignored
                    newdb.records[field] = values
            newdb.dictVarsToRecords()
            newdb.write(path)
    def db(self):
        fileType = self.data['dataFile'].rsplit('.')[-1].lower()
        self.fileType = fileType
        if fileType == 'csv':
            return csvreader.csv( self.data['dataFile'], 
                            -1, dType = DTYPE,
                            formatheader = FORMATHEADER,
                            numonly = 0 )
        elif fileType == 'dbf':
            return dbfreader.dbf2( self.data['dataFile'],
                            -1, dType = DTYPE,
                            formatheader = FORMATHEADER,
                            numonly = 1 ) 
        else:
            return None
    def loadWeights(self,db):
        wtFile = self.data['wtFiles'][self.data['wtFile']]
        f = open(wtFile,'r')
        dat = f.read()
        header = dat.splitlines()[0]
        f.close()
        if dat.count(',') > 5:
            sep = ','
            headline = 1
        else:
            sep = ' '
            headline = 1
        if len(header.split(sep))== 4:
            print "Reading idVar from file..."
            flag,n,file,idVar = header.split(sep)
            n = int(n)
            varIndex = db.varnames.index(idVar)
            idlist = db.records[varIndex]
            idlist = map(int,idlist)
            assert n == len(idlist)
        else:
            print "not reading from file, using record order..."
            idlist = db.idlist
        W = weights.spweight(
                idlist,
                wtFile,
                wtType ='binary',
                headline=headline,
                sep=sep,
                rowstand=1,
                power=1,
                dmax=0)
        return W
        
    def __dataFile(self,value=None):
        if value is not None:
            if os.path.exists(value) and not os.path.isdir(value):
                self.data['dataFile'] = value
                db = self.db()
                for v in self.newVars:
                    v.set('vars',db.numvarnames)
                self.set('vars',db.numvarnames)
            else:
                self.data['dataFile'] = False
    def __wtFile(self,value=None):
        if value is not None:
            if type(value) == int: #change the current wtFile from the GUI
                self.data['wtFile'] = value
            else: #add str to list
                if not value in self.data['wtFiles']:
                    self.data['wtFiles'].append(value)
                self.data['wtFile'] = self.data['wtFiles'].index(value)

class M_spLagVariable(AbstractModel):
    """ Model for an XRC panel that contains, [textCtrl] = W*[dropDown] """
    DATA_KEYS_ORDER = ['vars','var','newVarName','caution']
    def __init__(self):
        AbstractModel.__init__(self)
        #dispatch
        self.dispatch = {'vars':self.__vars}
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
        return self.data['newVarName'],self.data['vars'][self.data['var']]
    def get(self,key=None):
        if key:
            if key in self.data:
                return self.data[key]
            else:
                print "Model Error: %s, not in model"%key
                return None
        else:
            for key in self.data:
                if not key in self.DATA_KEYS_ORDER:
                    print "Model Warning: %s, not ORDERED"%key
                    self.DATA_KEYS_ORDER.append(key)
            for key in self.DATA_KEYS_ORDER:
                if not key in self.data:
                    print "Model Warning: %s, not in data"%key
                    self.data[key] = None
            return self.data
    def set(self,key=None,value=None,passive=False):
        print "M_spLagVar Setting",key,value
        if key in self.dispatch:
            self.dispatch[key](value)
        else:
            self.data[key] = value
        if not passive:
            self.update()
    def __vars(self,value=None):
        if value is not None:
            self.data['vars'] = value
            self.varsChanged=True

