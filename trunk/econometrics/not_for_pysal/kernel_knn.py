"""
KNN kernel weights. 
"""

__author__  = "Sergio J. Rey <srey@asu.edu>, David C. Folch <david.folch@asu.edu "

import pysal
from pysal.common import *
from pysal.weights import W


__all__ = ["knnW", "Kernel", "DistanceBand"]

class KernelKNN(W):
    """Spatial weights based on kernel functions where only weights for the K
    nearest neighbors are returned.
    
    ********************************************
    NOTE: This should be refactored and included
          in pysal.weights.Distance before 
          release 1.2
    ********************************************

    Parameters
    ----------

    data        : array (n,k)
                  n observations on k characteristics used to measure
                  distances between the n objects
    bandwidth   : float or array-like (optional)
                  the bandwidth :math:`h_i` for the kernel. 
    fixed       : binary
                  If true then :math:`h_i=h \\forall i`. If false then
                  bandwidth is adaptive across observations.
    k           : int
                  the number of nearest neighbors to use for determining
                  bandwidth. For fixed bandwidth, :math:`h_i=max(dknn) \\forall i`
                  where :math:`dknn` is a vector of k-nearest neighbor
                  distances (the distance to the kth nearest neighbor for each
                  observation).  For adaptive bandwidths, :math:`h_i=dknn_i`
    p           : float
                  Minkowski p-norm distance metric parameter:
                  1<=p<=infinity
                  2: Euclidean distance
                  1: Manhattan distance
    function    : string {'triangular','uniform','quadratic','quartic','gaussian'}
                  kernel function defined as follows with 

                  .. math::

                      z_{i,j} = d_{i,j}/h_i

                  triangular 

                  .. math::

                      K(z) = (1 - |z|) \ if |z| \le 1

                  uniform 

                  .. math::

                      K(z) = |z| \ if |z| \le 1

                  quadratic 

                  .. math::

                      K(z) = (3/4)(1-z^2) \ if |z| \le 1

                  quartic

                  .. math::

                      K(z) = (15/16)(1-z^2)^2 \ if |z| \le 1
                 
                  gaussian

                  .. math::

                      K(z) = (2\pi)^{(-1/2)} exp(-z^2 / 2)

    eps         : float
                  adjustment to ensure knn distance range is closed on the
                  knnth observations

    Examples
    --------

    >>> points=[(10, 10), (20, 10), (40, 10), (15, 20), (30, 20), (30, 30)]
    >>> kw=KernelKNN(points)
    >>> max([len(i) for i in kw.weights.values()]) == 3
    True
    >>> min([len(i) for i in kw.weights.values()]) == 3
    True
    >>> print kw.weights[0]
    [1.0, 0.50000004999999503, 0.44098306152674649]
    >>> kw.neighbors[0]
    [0, 1, 3]
    >>> print kw.bandwidth
    [[ 20.000002]
     [ 20.000002]
     [ 20.000002]
     [ 20.000002]
     [ 20.000002]
     [ 20.000002]]
    >>> kw_knn5=KernelKNN(points,bandwidth=15.0,k=5)
    >>> max([len(i) for i in kw_knn5.weights.values()]) == 6
    True
    >>> min([len(i) for i in kw_knn5.weights.values()]) == 6
    True
    >>> print kw_knn5[0]
    {0: 1.0, 1: 0.33333333333333337, 2: -1.0, 3: 0.2546440075000701, 4: -0.4907119849998598, 5: -0.88561808316412671}
    >>> print kw_knn5.neighbors[0]
    [0, 1, 3, 4, 5, 2]
    >>> print kw_knn5.bandwidth
    [[ 15.]
     [ 15.]
     [ 15.]
     [ 15.]
     [ 15.]
     [ 15.]]

    >>> kw15=KernelKNN(points,bandwidth=15.0)
    >>> max([len(i) for i in kw15.weights.values()]) == 3
    True
    >>> min([len(i) for i in kw15.weights.values()]) == 3
    True
    >>> print kw15[0]
    {0: 1.0, 1: 0.33333333333333337, 3: 0.2546440075000701}
    >>> print kw15.neighbors[0]
    [0, 1, 3]
    >>> print kw15.bandwidth
    [[ 15.]
     [ 15.]
     [ 15.]
     [ 15.]
     [ 15.]
     [ 15.]]

    Adaptive bandwidths user specified

    >>> bw=[25.0,15.0,25.0,16.0,14.5,25.0]
    >>> kwa=KernelKNN(points,bandwidth=bw)
    >>> max([len(i) for i in kwa.weights.values()]) == 3
    True
    >>> min([len(i) for i in kwa.weights.values()]) == 3
    True
    >>> print kwa.weights[0]
    [1.0, 0.59999999999999998, 0.55278640450004202]
    >>> print kwa.neighbors[0]
    [0, 1, 3]
    >>> print kwa.bandwidth
    [[ 25. ]
     [ 15. ]
     [ 25. ]
     [ 16. ]
     [ 14.5]
     [ 25. ]]

    Endogenous adaptive bandwidths 

    >>> kwea=KernelKNN(points,fixed=False)
    >>> max([len(i) for i in kwea.weights.values()]) == 3
    True
    >>> min([len(i) for i in kwea.weights.values()]) == 3
    True
    >>> print kwea.weights[0]
    [1.0, 0.10557289844279438, 9.9999990066379496e-08]
    >>> print kwea.neighbors[0]
    [0, 1, 3]
    >>> print kwea.bandwidth
    [[ 11.18034101]
     [ 11.18034101]
     [ 20.000002  ]
     [ 11.18034101]
     [ 14.14213704]
     [ 18.02775818]]

    Endogenous adaptive bandwidths with Gaussian kernel

    >>> kweag=KernelKNN(points,fixed=False,function='gaussian')
    >>> max([len(i) for i in kweag.weights.values()]) == 3
    True
    >>> min([len(i) for i in kweag.weights.values()]) == 3
    True
    >>> print kweag.weights[0]
    [0.3989422804014327, 0.26741902915776961, 0.24197074871621341]
    >>> print kweag.bandwidth
    [[ 11.18034101]
     [ 11.18034101]
     [ 20.000002  ]
     [ 11.18034101]
     [ 14.14213704]
     [ 18.02775818]]
    """

    def __init__(self,data,bandwidth=None,fixed=True,k=2, p=2,
                 function='triangular',eps=1.0000001,ids=None):

        # handle point_array
        if type(data).__name__=='ndarray':
            self.data=data
        elif type(data).__name__=='list':
            self.data=data
        else:
            raise Exception, 'Unsupported type'
        self.k=k+1 
        self.function=function.lower()
        self.fixed=fixed
        self.eps=eps
        self.kdt=KDTree(self.data)
        self.dmat,self.neigh=self.kdt.query(data,k=k+1,p=p)
        if bandwidth:
            try:
                bandwidth=np.array(bandwidth)
                bandwidth.shape=(len(bandwidth),1)
            except:
                bandwidth=np.ones((len(data),1),'float')*bandwidth
            self.bandwidth=bandwidth
        else:
            self._set_bw()

        self._eval_kernel()
        neighbors, weights = self._k_to_W(ids)
        W.__init__(self,neighbors,weights,ids)

    def _k_to_W(self, ids=None):
        allneighbors={}
        weights={}
        if ids:
            ids = np.array(ids)
        else:
            ids = np.arange(len(self.data))
        for i, neighbors in enumerate(self.kernel):
            if len(self.neigh[i]) == 0:
                allneighbors[ids[i]] = []
                weights[ids[i]] = []
            else:    
                allneighbors[ids[i]] = list(ids[self.neigh[i]])
                weights[ids[i]] = self.kernel[i].tolist()
        return allneighbors,weights

    def _set_bw(self):
        if self.fixed:
            # use max knn distance as bandwidth
            bandwidth=self.dmat.max()*self.eps
            n=len(self.dmat)
            self.bandwidth=np.ones((n,1),'float')*bandwidth
        else:
            # use local max knn distance
            self.bandwidth=self.dmat.max(axis=1)*self.eps
            self.bandwidth.shape=(self.bandwidth.size,1)

    def _eval_kernel(self):
        # get distances for neighbors
        data=np.array(self.data)
        bw=self.bandwidth
        z=[]
        for i,dists in enumerate(self.dmat):
            zi=dists/bw[i]
            z.append(zi)
        zs=z
        # functions follow Anselin and Rey (2010) table 5.4
        if self.function=='triangular':
            self.kernel=[1-z for z in zs]
        elif self.function=='uniform':
            self.kernel=z
        elif self.function=='quadratic':
            self.kernel=[(3./4)*(1-z**2) for z in zs]
        elif self.function=='epanechnikov':
            self.kernel=[(1-z**2) for z in zs]
        elif self.function=='quartic':
            self.kernel=[(15./16)*(1-z**2)**2 for z in zs]
        elif self.function=='bisquare':
            self.kernel=[(1-z**2)**2 for z in zs]
        elif self.function=='gaussian':
            c=np.pi*2
            c=c**(-0.5)
            self.kernel=[c*np.exp(-(z**2)/2.) for z in zs]
        else:
            raise Exception, 'Unsupported kernel function',self.function
        

def _test():
    """Doc test"""
    import doctest
    doctest.testmod()

if __name__ == "__main__":
    _test()
    
