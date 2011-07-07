import types
import inspect
DEBUG = False

class AbstractModel(object):
    """ From wxPIA/Chapter-05 """
    def __init__(self):
        self._modelData = {}
        self.listeners = []

    def addListener(self, listenerFunc):
        self.listeners.append(listenerFunc)

    def removeListener(self, listenerFunc):
        self.listeners.remove(listenerFunc)

    def update(self,tag=None):
        if tag:
            if DEBUG: print "Updating: ",tag
        for eachFunc in self.listeners:
            eachFunc(tag)

    ### Custom Below here
    TAGS = []
    def __iter__(self):
        return self.next()
    def next(self):
        tags = self.TAGS + [k for k in self._modelData if k not in self.TAGS]
        for key in tags:
            yield key,self.__getattribute__(key)
    def getByTag(self,tag):
        return self.__getattribute__(tag)
    @staticmethod
    def abstractProp(tag,typ=None,updateAll=False):
        if not issubclass(type(tag),basestring):
            return None
        def fget(self,tag=tag):
            if tag in self._modelData:
                return self._modelData[tag]
            else:
                return ''
        def fset(self,value,tag=tag,typ=typ,updateAll=updateAll):
            if typ and type(value) != typ:
                try:
                    if value == None:
                        value = None
                    else:
                        value = typ(value)
                except:
                    raise TypeError, "The value '%r' could not be cast as type '%r'"%(value,typ)
            if self._modelData.get(tag,'') != value: # Only update on a change!
                self._modelData[tag] = value
                if updateAll:
                    self.update()
                else:
                    self.update(tag)
        def fdel(self,tag=tag):
            if tag in self._modelData:
                del self._modelData[tag]
        del tag
        del typ
        del updateAll
        return property(**locals())

def remapEvtsToDispatcher(instance,dispatcherMethod):
    """ The new version of XRCED takes care of evt bindings for us,
        it also adds an evt handler for each binding,
        This function remaps those events to a dispatcher
    """
    ## identify binds and remap to dispatcher
    for name in dir(instance):
        # scan the parent class for event bindings
        if name[:2]=='On' and '_' in name:
            obj = getattr(instance,name) #grab the object from the instance
            #make sure it is an instancemethod and has the right argSpec
            if type(obj) == types.MethodType and inspect.getargspec(obj)[0] == ['self', 'evt']:
                if DEBUG: print "remapping instance.%s(evt) to instance.dispatch('%s',evt)"%(name,name)
                setattr(instance,name,evt(dispatcherMethod,name))
class evt:
    """This class redirects traditional event handlers to a dispatcher"""
    def __init__(self,dispatcherMethod,name):
        self.name = name
        self.dispatcherMethod = dispatcherMethod
    def __call__(self,evt):
        """pass the evt and the name of the evt to the dispatcher"""
        self.dispatcherMethod(self.name,evt)

