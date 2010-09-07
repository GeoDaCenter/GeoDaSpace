import numpy as np
from scipy import sparse as SP
import pysal
from spHetErr import Spatial_Error_Het

class spFilter:
    '''
    computer the spatially filtered variables
    
    Parameters
    ----------

    x       : array
              nxk array of independent variables (assumed to be aligned with y)
    y       : array
              nx1 array of dependent variable
    w       : weight
              PySAL weights instance  
    lamda   : double
              spatial autoregressive parameter

    Attributes
    ----------

    x       : array
              nxk array of independent variables (assumed to be aligned with y)
    y       : array
              nx1 array of dependent variable
    w       : weight
              PySAL weights instance 
    lamb    : double
              spatial autoregressive parameter
    z       : array
              n*(k+1) array of [x, w*y]
    w_matrix: sparse matrix of w
    
    Methods
    ----------

    get_spFilter: return spatially filtered variables
    
    get_S       : return sparse matrix of w
    
    Examples
    --------

    >>> import numpy as np
    >>> import pysal
    >>> db=pysal.open("examples/columbus.dbf","r")
    >>> y = np.array(db.by_col("CRIME"))
    >>> y = np.reshape(y, (49,1))
    >>> X = []
    >>> X.append(db.by_col("INC"))
    >>> X.append(db.by_col("HOVAL"))
    >>> X = np.array(X).T
    >>> w=pysal.open("examples/columbus.GAL").read()  
    >>> solu = spFilter(X,y,w,0.5)
    >>> print solu.ys 
    >>>
    [[ -15.7812905]
    [ -32.158758 ]
    [ -32.4865705]
    [ -27.410798 ]
    [ -97.3822125]
    [  -3.9670885]
    [ -91.2635635]
    [ -99.43012  ]
    [-118.107001 ]
    [ -35.0421145]
    [ -35.2276465]
    [ -77.3782495]
    [ -35.4055665]
    [ -59.8451955]
    [ -75.117567 ]
    [-140.013057 ]
    [ -33.4549225]
    [ -29.5261305]
    [ -27.6813485]
    [-177.105114 ]
    [ -19.001647 ]
    [-119.702461 ]
    [ -42.9538575]
    [ -88.131421 ]
    [-132.2616395]
    [ -98.780399 ]
    [ -19.629096 ]
    [-118.43816  ]
    [ -92.400958 ]
    [ -60.3964295]
    [ -31.2535   ]
    [ -37.7388985]
    [ -44.0567675]
    [ -42.005636 ]
    [ -69.722601 ]
    [ -55.2186525]
    [ -62.0116055]
    [ -72.6139295]
    [ -25.6059015]
    [ -53.5011755]
    [ -24.8541415]
    [ -24.3181745]
    [ -40.9453225]
    [ -14.9928235]
    [ -22.078858 ]
    [ -12.8126545]
    [  10.124343 ]
    [  -9.115376 ]
    [ -11.508765 ]]
    '''
 
    def __init__(self,x,y,w,lamb):
        self.x = x
        self.y = y
        self.w = w
        self.lamb = lamb
        
        # convert w into sparse matrix      
        self.w_matrix = self.get_S(w)
        
        self.ys = self.get_spFilter(y)
        self.xs = self.get_spFilter(x)
        yl = self.w_matrix.dot(y,)
        self.wys = self.get_spFilter(yl)
        self.z = np.hstack((x,yl))
        self.zs = self.get_spFilter(self.z)
        
        
    def get_spFilter(self, sf):
        '''
        return spatially filtered variable
        
        Parameters
        ----------

        sf              : y, x, wy or z
                      

        Returns
        -------

        sp              : array
                          
        '''
        
        sp = sf - self.lamb * self.w_matrix.dot(sf,)
        
        return sp
        
    
    # copy from spHetErr.py
    def get_S(self,w):
        """
        Converts pysal W to scipy csr_matrix
        ...

        Parameters
        ----------

        w               : W
                      Spatial weights instance

        Returns
        -------

        Implicit        : csr_matrix
                      PySAL W object converted into Scipy sparse matrix
                
        """
        data = []
        indptr = [0]
        indices = []
        for ob in w.id_order:
            data.extend(w.weights[ob])
            indptr.append(indptr[-1] + len(w.weights[ob]))
            indices.extend(w.neighbors[ob])
        data = np.array(data)
        indices = np.array(indices)
        indptr = np.array(indptr)
        
        return SP.csr_matrix((data,indices,indptr),shape=(w.n,w.n))

def _test():
    import doctest
    doctest.testmod()
    
if __name__ == '__main__':
    _test()
    '''
    x = [[1,2,3,4],
         [1,2,3,4],
         [1,2,3,4],
         [1,2,3,4]]
    x = np.array(x)
    y = [[1],
         [2],
         [3],
         [4]]
    y = np.array(y)
    neighbors = {0:[0,2,3],1:[1,3],2:[0,2],3:[0,1,3]}
    w = pysal.W(neighbors)
    #w=pysal.open("../examples/columbus.GAL").read()  
    solu = spFilter(x,y,w,0.5)
    '''
    '''
    db=pysal.open("examples/columbus.dbf","r")
    y = np.array(db.by_col("CRIME"))
    y = np.reshape(y, (49,1))
    X = []
    X.append(db.by_col("INC"))
    X.append(db.by_col("HOVAL"))
    X = np.array(X).T
    w=pysal.open("examples/columbus.GAL").read()  
    solu = spFilter(X,y,w,0.5)
    print solu.ys
    print solu.ys.shape
    #print solu.xs
    #print solu.wys
    #print solu.zs
    '''
        
    