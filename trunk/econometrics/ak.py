from scipy.stats.stats import chisqprob
from scipy.stats import norm
from pysal.spreg.diagnostics_sp import lmErr
import numpy as np
import numpy.linalg as la
import pysal


from pysal.spreg.diagnostics_sp import spDcache, get_mI

class AKtest:
    """
    Moran's I test of spatial autocorrelation for IV estimation.
    Implemented following the original reference Anselin and Kelejian
    (1997) [1]_
    ...

    Parameters
    ----------

    iv          : TSLS
                  Regression object from TSLS class
    w           : W
                  Spatial weights instance 
    case        : string
                  Flag for special cases (default to 'nosp'):
                    * 'nosp': Only NO spatial end. reg.
                    * 'gen': General case (spatial lag + end. reg.)

    Attributes
    ----------

    mi          : float
                  Moran's I statistic for IV residuals
    ak          : float
                  Square of corrected Moran's I for residuals::
                    
                  .. math::

                        ak = \dfrac{N \times I^*}{\phi^2}

                  Note: if case='nosp' then it simplifies to the LMerror
    p           : float
                  P-value of the test

    References
    ----------

    .. [1] Anselin, L., Kelejian, H. (1997) "Testing for spatial error
    autocorrelation in the presence of endogenous regressors". Interregional
    Regional Science Review, 20, 1.

    .. [2] Kelejian, H.H., Prucha, I.R. and Yuzefovich, Y. (2004)
    "Instrumental variable estimation of a spatial autorgressive model with
    autoregressive disturbances: large and small sample results". Advances in
    Econometrics, 18, 163-198.
    
    Examples
    --------

    We first need to import the needed modules. Numpy is needed to convert the
    data we read into arrays that ``spreg`` understands and ``pysal`` to
    perform all the analysis. The TSLS is required to run the model on
    which we will perform the tests.

    >>> import numpy as np
    >>> import pysal
    >>> from twosls import BaseTSLS as TSLS
    >>> from twosls_sp import GM_Lag

    Open data on Columbus neighborhood crime (49 areas) using pysal.open().
    This is the DBF associated with the Columbus shapefile.  Note that
    pysal.open() also reads data in CSV format; since the actual class
    requires data to be passed in as numpy arrays, the user can read their
    data in using any method.  

    >>> db = pysal.open(pysal.examples.get_path("columbus.dbf"),'r')
    
    Before being able to apply the diagnostics, we have to run a model and,
    for that, we need the input variables. Extract the CRIME column (crime
    rates) from the DBF file and make it the dependent variable for the
    regression. Note that PySAL requires this to be an numpy array of shape
    (n, 1) as opposed to the also common shape of (n, ) that other packages
    accept.

    >>> y = np.array(db.by_col("CRIME"))
    >>> y = np.reshape(y, (49,1))

    Extract INC (income) vector from the DBF to be used as
    independent variables in the regression.  Note that PySAL requires this to
    be an nxj numpy array, where j is the number of independent variables (not
    including a constant). By default this model adds a vector of ones to the
    independent variables passed in, but this can be overridden by passing
    constant=False.

    >>> X = []
    >>> X.append(db.by_col("INC"))
    >>> X = np.array(X).T

    In this case, we consider HOVAL (home value) as an endogenous regressor,
    so we acknowledge that by reading it in a different category.

    >>> yd = []
    >>> yd.append(db.by_col("HOVAL"))
    >>> yd = np.array(yd).T

    In order to properly account for the endogeneity, we have to pass in the
    instruments. Let us consider DISCBD (distance to the CBD) is a good one:

    >>> q = []
    >>> q.append(db.by_col("DISCBD"))
    >>> q = np.array(q).T

    Now we are good to run the model. It is an easy one line task.

    >>> reg = TSLS(y, X, yd, q=q)

    Now we are concerned with whether our non-spatial model presents spatial
    autocorrelation in the residuals. To assess this possibility, we can run
    the Anselin-Kelejian test, which is a version of the classical LM error
    test adapted for the case of residuals from an instrumental variables (IV)
    regression. First we need an extra object, the weights matrix, which
    includes the spatial configuration of the observations
    into the error component of the model. To do that, we can open an already
    existing gal file or create a new one. In this case, we will create one
    from ``columbus.shp``.

    >>> w = pysal.rook_from_shapefile(pysal.examples.get_path("columbus.shp")) 

    Unless there is a good reason not to do it, the weights have to be
    row-standardized so every row of the matrix sums to one. Among other
    things, this allows to interpret the spatial lag of a variable as the
    average value of the neighboring observations. In PySAL, this can be
    easily performed in the following way:

    >>> w.transform = 'r'

    We are good to run the test. It is a very simple task:

    >>> ak = AKtest(reg, w)

    And explore the information obtained:

    >>> print('AK test: %f\tP-value: %f'%(ak.ak, ak.p))
    AK test: 4.642895      P-value: 0.031182

    The test also accomodates the case when the residuals come from an IV
    regression that includes a spatial lag of the dependent variable. The only
    requirement needed is to modify the ``case`` parameter when we call
    ``AKtest``. First, let us run a spatial lag model:

    >>> reg_lag = GM_Lag(y, X, yd, q=q, w=w)

    And now we can run the AK test and obtain similar information as in the
    non-spatial model.

    >>> ak_sp = AKtest(reg, w, case='gen')
    >>> print('AK test: %f\tP-value: %f'%(ak_sp.ak, ak_sp.p))
    AK test: 1.157593      P-value: 0.281965
    
            """

    def __init__(self, iv, w, case='nosp'):
        if case == 'gen':
            cache = spDcache(iv, w)
            self.mi, self.ak, self.p = akTest(iv, w, cache)
        elif case == 'nosp':
            cache = spDcache(iv, w)
            self.mi = get_mI(iv, w, cache)
            self.ak, self.p = lmErr(iv, w, cache)
        else:
            print """\n
            Fix the optional argument 'case' to match the requirements:
                * 'gen': General case (spatial lag + end. reg.)
                * 'nosp': No spatial end. reg.
            \n"""

def akTest(iv, w, spDcache):
    """
    Computes AK-test for the general case (end. reg. + sp. lag)
    ...

    Parameters
    ----------

    iv          : STSLS_dev
                  Instance from spatial 2SLS regression
    w           : W
                  Spatial weights instance 
   spDcache     : spDcache
                  Instance of spDcache class

    Attributes
    ----------
    mi          : float
                  Moran's I statistic for IV residuals
    ak          : float
                  Square of corrected Moran's I for residuals::
                    
                  .. math::

                        ak = \dfrac{N \times I^*}{\phi^2}

    p           : float
                  P-value of the test

    ToDo:
        * Code in as Nancy
        * Compare both
    """
    mi = get_mI(iv, w, spDcache)
    # Phi2
    etwz = np.dot(iv.u.T, (w.sparse * iv.z))
    a = np.dot(etwz,np.dot(iv.varb,etwz.T))
    s12 = (w.s0 / w.n)**2
    phi2 = ( spDcache.t + (4.0 / iv.sig2n) * a ) / (s12 * w.n)
    ak = w.n * mi**2 / phi2
    pval = chisqprob(ak, 1)
    return (mi, ak[0][0], pval[0][0])


def _test():
    import doctest
    doctest.testmod()

                     
if __name__ == '__main__':
    _test()    

