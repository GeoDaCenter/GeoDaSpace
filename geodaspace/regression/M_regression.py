import os.path
import csv

from geodaspace import abstractmodel
import pysal
import numpy as np
from econometrics.gs_dispatcher import spmodel
from geodaspace.preferences.model import preferencesModel

class GeoDaSpace_W_Obj(object):
    """ A wrapper around W objs to carry meta data like name, state, etc """
    def __init__(self, obj = None):
        self._enabled = True
        if issubclass(type(obj),pysal.W):
            self._w_obj = obj
        else:
            raise TypeError, "obj must be of type W."
    def __eq__(self,other):
        if not issubclass(type(other),GeoDaSpace_W_Obj):
            return False
        elif other.has_meta and self.has_meta:
            return other.w.meta == self.w.meta
        elif hasattr(self,'_path') and hasattr(other,'_path'):
            return other._path == self._path
        else:
            return self.w == other.w
    def __get_enabled(self):
        return self._enabled
    def __set_enabled(self,v):
        self._enabled = bool(v)
    enabled = property(__get_enabled, __set_enabled)
    @property
    def saved(self):
        if hasattr(self,'_path'):
            return True
    @property
    def w(self):
        return self._w_obj
    @classmethod
    def from_path(cls,path):
        """
        Instantiates an instance of GeoDaSpace_W_Obj from a path
        """
        w = cls(pysal.open(path,'r').read())
        w._path = path
        return w
    @property
    def has_meta(self):
        return hasattr(self.w,'meta')
    @property
    def name(self):
        if self.has_meta:
            if self.w.meta['shape file'] == 'unknown':
                name = 'File'
            else:
                name = os.path.basename(self.w.meta['shape file'])
            name+= ': '+self.w.meta['method']
            if 'method options' in self.w.meta:
                opts = self.w.meta['method options']
                if type(opts) == list:
                    name+= ': '+', '.join(map(str,self.w.meta['method options']))
                else:
                    name+= ': '+opts
        elif hasattr(self,'_path'):
            name = os.path.basename(self._path)
        else:
            name = "weights object"
        return name

class guiRegModel(abstractmodel.AbstractModel):
    STATE_EMPTY = 0
    STATE_CHANGED = 1
    STATE_SAVED = 2
    def __init__(self):
        abstractmodel.AbstractModel.__init__(self)
        self.fileObj = None
        self.reset()
    def update(self):
        self.state = self.STATE_CHANGED
        abstractmodel.AbstractModel.update(self)
    # The Setters
    def setDataFile(self,path,passive=False):
        if self.data['fname'] == path:
            self.loadFieldNames()
            self.update()
        else:
            if not passive:
                self.reset()
            self.data['fname'] = path
            self.loadFieldNames()
            self.update()
    def setSpec(self,spec):
        self.data['spec'] = spec
        self.update()
    def setModelType(self,setup):
        self.data['modelType'] = setup
        self.update()
    def addMWeightsFile(self,path=None,obj=None):
        if obj:
            obj = GeoDaSpace_W_Obj(obj)
            if obj not in self.data['mWeights']:
                self.data['mWeights'].append(obj)
                self.update()
        elif path:
            obj = GeoDaSpace_W_Obj.from_path(path)
            if obj not in self.data['mWeights']:
                self.data['mWeights'].append(obj)
                self.update()
        else:
            pass
    def addKWeightsFile(self,path=None,obj=None):
        if obj:
            obj = GeoDaSpace_W_Obj(obj)
            if obj not in self.data['kWeights']:
                self.data['kWeights'].append(obj)
                self.update()
        elif path:
            obj = GeoDaSpace_W_Obj.from_path(path)
            if obj not in self.data['kWeights']:
                self.data['kWeights'].append(obj)
                self.update()
        else:
            pass
    def checkKW(self,obj):
        print "running checkKW() this function needs work."
        avgN = sum(obj.cardinalities.values())/float(obj.n)
        if avgN < 10.0:
            return False
        return True

    # The Getters
    def getVariables(self):
        if 'db' in self.data:
            if 'header' in self.data['db']:
                return self.data['db']['header']
        else:
            return []
    def getDataFile(self):
        if 'fname' in self.data:
            return os.path.basename(self.data['fname'])
        else:
            return ''
    def getSpec(self):
        return self.data['spec']
    def getModelType(self):
        return self.data['modelType']
    def getMWeightsFiles(self):
        return self.data['mWeights']
    def getMWeightsEnabled(self):
        return [w for w in self.data['mWeights'] if w.enabled]
    def removeMW(self,idx):
        w = self.data['mWeights'].pop(idx)
        self.update()
        return w
    def removeKW(self,idx):
        w = self.data['kWeights'].pop(idx)
        self.update()
        return  w
    def getKWeightsFiles(self):
        return self.data['kWeights']
    def getModelMethod(self):
        raise DeprecationWarning,'getModelMethod has been depracted'
        mType = self.data['modelType']['mType']
        #bEndo = self.data['modelType']['endogenous'] # Endo is not a checkbox in the GUI
        if mType == 0: #Standard
            space = None
            #allowed estimators.
            #est =[OLS ,IV  ,GMM   ]
            if bEndo:
                est = [False,True,False]
            else:
                est = [True,False,False]
            meth = 'ols'
        elif mType == 1: #Spatial Lag
            space = 'lag'
            est = [False,True,False]
            meth = 'twosls'
        elif mType == 2: #Spatial Error
            space = 'error'
            est = [True,False,True]
            meth = 'ols'
        elif mType == 3: #Spatial Lag+Error
            space = 'lagerror'
            est = [False,True,True]
            meth = 'twosls'
        elif mType == 4: #Regimes
            space = None
            est = None
            meth = None
        return (space,est,meth)

    # Utility Functions
    def reset(self):
        self.state = self.STATE_EMPTY
        self.fileObj = None
        self.data = {}
        self.data['fname'] = ''
        self.data['dType'] = 'listvars' #listvars: list of lists by variable
        self.data['formatheader'] = 0
        self.data['numonly'] = 0
        self.data['spec'] = {'y':"",'YE':[],'H':[],'R':"",'S':"",'T':"",'X':[]}
        self.data['mWeights'] = []
        self.data['kWeights'] = []
        self.data['modelType'] = {'mType':0,'endogenous':False,'method':0}
        self.data['modelType']['error'] = {'classic':True,'white':False,'hac':False,'het':False}
        self.data['modelType']['spatial_tests'] = {'lm':True}
        config = {}
        config.update(preferencesModel.DEFAULTS)
        self.data['config'] = config
    def loadFieldNames(self):
        """Sets the db with field names"""
        self.data['db'] = self.db(headerOnly=True)
    def db(self,headerOnly=False):
        if 'fname' in self.data:
            fileType = self.data['fname'].rsplit('.')[-1].lower()
            self.fileType = fileType
            if fileType == 'csv':
                if headerOnly:
                    f = pysal.open(self.data['fname'],'r')
                    db = {}
                    db['header'] = f.header
                    f.close()
                    return db
                else:
                    return pysal.open(self.data['fname'],'r')
            elif fileType == 'dbf':
                db = pysal.open( self.data['fname'] , 'r')
                header = []
                # grab only the numeric fields.
                if headerOnly:
                    for field,spec in zip(db.header, db.field_spec):
                        typ = spec[0].lower()
                        if typ in  ['d','n']:
                            header.append(field)
                    return {'header':header}
                else:
                    return db
            else:
                print "Unknown File Type"
                return False
        else:
            return None
    
#    def prepWeights(self, db=None):
#        """Prepares the weight file dictionaries"""
#        if not db:
#            db = self.db()
#        data = self.data
#
#        mweights = []
#        for fname in data['mWeights']:
#            f = open(fname,'r')
#            dat = f.read()
#            header = dat.splitlines()[0]
#            f.close()
#            if dat.count(',') > 5:
#                sep = ','
#                headline = 1
#            else:
#                sep = ' '
#                headline = 1
#            if len(header.split(sep))== 4:
#                print "Reading idVar from file..."
#                flag,n,file,idVar = header.split(sep)
#                n = int(n)
#                varIndex = db.varnames.index(idVar)
#                idlist = db.records[varIndex]
#                idlist = map(int,idlist)
#                assert n == len(idlist)
#            else:
#                print "not reading from file, using record order..."
#                idlist = db.idlist
#            mw = weights.spweight(
#                idlist,
#                fname,
#                wtType ='binary',
#                headline = headline,
#                sep = sep,
#                rowstand = 1,
#                power = 1,
#                dmax = 0 )
#            mweights.append(mw)
#
#        wtTypes=['epanech','triangle']
#        kweights = []
#        fname = None
#        for fname in data['kWeights']:
#            f = open(fname,'r')
#            dat = f.read()
#            header = dat.splitlines()[0]
#            f.close()
#            if dat.count(',') > 5:
#                sep = ','
#                headline = 1
#            else:
#                sep = None #white space
#                headline = 1
#            if len(header.split(sep))== 4:
#                print "Reading idVar from file..."
#                print header
#                flag,n,file,idVar = header.split(sep)
#                n = int(n)
#                varIndex = db.varnames.index(idVar)
#                idlist = db.records[varIndex]
#                idlist = map(int,idlist)
#                assert n == len(idlist)
#            else:
#                print "not reading from file, using record order..."
#                idlist = db.idlist
#            for wt in wtTypes:
#                kw = weights.spweight(
#                    idlist,
#                    fname,
#                    wtType = wt,
#                    headline = headline,
#                    sep = sep,
#                    rowstand = 0,
#                    power = 1,
#                    dmax = 0 )
#                kweights.append(kw)
#
#        if not mweights: mweights.append([])
#        if not kweights: kweights.append([])
#        return (mweights,kweights)
        
    def save(self,fileObj):
        """ Returns the contents of the model """
        self.state = self.STATE_SAVED
        fileObj.write(str(self.data))
        fileObj.flush()
        self.fileObj = fileObj
    def open(self,s):
        """ Loads the contents of the model from s """
        try:
            self.reset()
            self.data = eval(s)
            if not os.path.exists(self.data['fname']):
                return False
            self.loadFieldNames()
            self.update()
            self.state = self.STATE_SAVED
            return True
        except:
            raise TypeError,"The Supplied Model File Was Invalid."
    def verify(self):
        #if self.data['modelType']['endogenous'] == True: #endogenous == yes
        lYE,lH = len(self.data['spec']['YE']),len(self.data['spec']['H'])
        if lYE > 0 or lH >0:
            if lH < lYE:
                return False,"There need to be at least as many instruments (H) as endogenous variables (YE)."
            if lYE == 0:
                return False,'Please add endogenous variables (YE) or disable the "Endogeneity" option.'
        if self.data['spec']['y'] and self.data['spec']['X']:
            pass
        else:
            return False,'Model Spec is incomplete.  Please populate both X and Y'
        if self.data['modelType']['spatial_tests']['lm'] and not self.getMWeightsEnabled():
            return False,'LM Test requires Model Weights, please add or create a weights file, or disable "LM".'
        return True,None
    def run(self,path=None, predy_resid=None):
        """ Runs the Model """
        print self.verify()
        if not self.verify()[0]:
            return False
        data = self.data
        print data

        # Build up args for dispatcher
        # weights
        w_names = [w.name for w in data['mWeights'] if w.enabled]
        w_list = [w.w for w in data['mWeights'] if w.enabled]
        for w,name in zip(w_list,w_names):
            w.name = name
        wk_names = [w.name for w in data['kWeights'] if w.enabled]
        wk_list = [w.w for w in data['kWeights'] if w.enabled]
        for w,name in zip(wk_list,wk_names):
            w.name = name
        db = pysal.open( data['fname'] ,'r')
        # y
        name_y = data['spec']['y']
        y = np.array([db.by_col(name_y)]).T

        # x
        x = []
        x_names = data['spec']['X']
        for x_name in x_names:
            x.append(db.by_col(x_name))
        x = np.array(x).T

        # YE
        ye = []
        ye_names = data['spec']['YE']
        for ye_name in ye_names:
            ye.append(db.by_col(ye_name))
        ye = np.array(ye).T
       
        # H
        h = []
        h_names = data['spec']['H']
        for h_name in h_names:
            h.append(db.by_col(h_name))
        h = np.array(h).T

        mtypes = {0: 'Standard', 1: 'Spatial Lag', 2: 'Spatial Error', 3: 'Spatial Lag+Error'}
        model_type = mtypes[data['modelType']['mType']]

        # These options are not available yet....
        r = None
        name_r = None
        s = None
        name_s = None
        t = None
        name_t = None
        sig2n_k_ols = False
        sig2n_k_tsls = False
        sig2n_k_gmlag = False
        config = data['config']

        print w_list,wk_list
        fname = os.path.basename(data['fname'])
        results = spmodel(fname, w_list, wk_list, y, name_y, x, x_names, ye, ye_names,
                 h, h_names, r, name_r, s, name_s, t, name_t,
                 model_type, #data['modelType']['endogenous'],
                 data['modelType']['spatial_tests']['lm'],
                 data['modelType']['error']['white'], data['modelType']['error']['hac'], data['modelType']['error']['het'],
                 #config.....
                 config['sig2n_k_ols'], config['sig2n_k_2sls'], config['sig2n_k_gmlag'],
                 config['gmm_max_iter'], config['gmm_epsilon'], config['gmm_inferenceOnLambda'], config['gmm_inv_method'], config['gmm_step1c'],
                 config['instruments_w_lags'], config['instruments_lag_q'],
                 config['output_vm_summary'], predy_resid,
                 config['other_ols_diagnostics'], config['other_residualMoran']
                 )
        print results
        for r in results:
            path.write(r.summary)
            path.write('\n\n\n')

        return [r.summary for r in results]
#        modelHeader = 'ModelType = %(mtype)s, Endogenous = %(endo)s'#, Standard Errors = %(errors)s'
#        modelSpec = {}
#        errors = []
#        if data['modelType']['error']['hac']:
#            errors.append('HAC')
#        if data['modelType']['error']['het']:
#            errors.append('KP HET')
#        if data['modelType']['error']['white']:
#            errors.append('White')
#        if errors:
#            modelSpec['errors'] = ', '.join(errors)
#        else:
#            modelSpec['errors'] = 'None'
#        if data['modelType']['endogenous'] == True: #endogenous == yes
#            modelSpec['endo'] = 'Y'
#        else: 
#            modelSpec['endo'] = 'N'
#        
#        db = self.db()
#        if not db:
#            raise TypeError, "This filetype is not yet supported"
#        
#        # Setup the weights objects
#        mws, kws = self.prepWeights(db)
#
#        # The Sprecification is stored in a dictionary object
#        spec = data['spec']
#        spec['yend'] = spec['YE']
#
#        if data['modelType']['endogenous'] == True: #endogenous == yes
#            endo = 'endog'
#        else: #endogenous == no
#            endo = 'classic'
#
#        option = ''
#        if data['modelType']['error']['hac']:
#            option = 'hac'
#        if data['modelType']['error']['het']:
#            option = 'het'
#
#        mType = data['modelType']['mType']
#        if mType == 0: #Standard
#            modelSpec['mtype'] = 'Standard'
#            space = ''
#            meth = 'ols'
#            if endo=='endog': #LINE ADDED BY NANCY
#                meth= 'twosls'  #LINE ADDED BY NANCY
#                
#        elif mType == 1: #Spatial Lag
#            modelSpec['mtype'] = 'Spatial Lag'
#            space = 'lag'
#            meth = 'twosls'
#        elif mType == 2: #Spatial Error
#            modelSpec['mtype'] = 'Spatial Error'
#            space = 'error'
#            meth = 'ols'
#            if endo=='endog': #LINE ADDED BY NANCY
#                meth= 'twosls'  #LINE ADDED BY NANCY
#                
#        elif mType == 3: #Spatial Lag+Error
#            modelSpec['mtype'] = 'Spatial Lag+Error'
#            space = 'lagerror'
#            meth = 'twosls'
#        elif mType == 4: #Regimes
#            modelSpec['mtype'] = 'Regimes'
#            pass
#
#        mdls = []
#        if option is not 'hac':
#            kws = [[]]
#        else: #has is on...
#            for mw in mws: #lets runs all the None HAC models first....
#                mdl = spreg.spmodel(db, spec, model=endo, mweights = mw, kweights = [], space = space, option = '', white=data['modelType']['error']['white'],extraHeader=modelHeader%modelSpec, lmtestOn=data['modelType']['spatial_tests']['lm'])
#                if meth =='ols':
#                    mdl.ols(outfile=path)
#                elif meth == 'twosls':
#                    mdl.twosls(outfile=path)
#                mdls.append(mdl)
#        for mw in mws:
#            for kw in kws:
#                mdl = spreg.spmodel(db, spec, model=endo, mweights = mw, kweights = kw, space = space, option = option, white=data['modelType']['error']['white'],extraHeader=modelHeader%modelSpec, lmtestOn=data['modelType']['spatial_tests']['lm'])
#                if meth =='ols':
#                    mdl.ols(outfile=path)
#                elif meth == 'twosls':
#                    mdl.twosls(outfile=path)
#                mdls.append(mdl)
#        return mdls
