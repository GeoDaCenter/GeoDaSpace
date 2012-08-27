import numpy as np
import regimes as REGI
import user_output as USER
from utils import sphstack
from ols import BaseOLS
from twosls import BaseTSLS
import summary_output as SUMMARY

class TSLS_Regimes(BaseTSLS, REGI.Regimes_Frame):
    """
    Examples
    --------

We first need to import the needed modules, namely numpy to convert the
    data we read into arrays that ``spreg`` understands and ``pysal`` to
    perform all the analysis.

    >>> import numpy as np
    >>> import pysal

    Open data on Columbus neighborhood crime (49 areas) using pysal.open().
    This is the DBF associated with the Columbus shapefile.  Note that
    pysal.open() also reads data in CSV format; since the actual class
    requires data to be passed in as numpy arrays, the user can read their
    data in using any method.  

    >>> db = pysal.open(pysal.examples.get_path("columbus.dbf"),'r')
 
    Extract the CRIME column (crime rates) from the DBF file and make it the
    dependent variable for the regression. Note that PySAL requires this to be
    an numpy array of shape (n, 1) as opposed to the also common shape of (n, )
    that other packages accept.

    >>> y_var = 'CRIME'
    >>> y = np.array([db.by_col(y_var)]).reshape(49,1)

    Extract INC (income) vector from the DBF to be used as
    independent variables in the regression. Other variables can be inserted
    by adding their names to x_var, such as x_var = ['Var1','Var2','...]
    Note that PySAL requires this to be an nxj numpy array, where j is the
    number of independent variables (not including a constant). By default
    this model adds a vector of ones to the independent variables passed in,
    but this can be overridden by passing constant=False.

    >>> x_var = ['INC']
    >>> x = np.array([db.by_col(name) for name in x_var]).T

    In this case we consider HOVAL (home value) is an endogenous regressor.
    We tell the model that this is so by passing it in a different parameter
    from the exogenous variables (x).

    >>> yd_var = ['HOVAL']
    >>> yd = np.array([db.by_col(name) for name in yd_var]).T

    Because we have endogenous variables, to obtain a correct estimate of the
    model, we need to instrument for HOVAL. We use DISCBD (distance to the
    CBD) for this and hence put it in the instruments parameter, 'q'.

    >>> q_var = ['DISCBD']
    >>> q = np.array([db.by_col(name) for name in q_var]).T

    The different regimes in this data are given according to the North and 
    South dummy (NSA).

    >>> regimes = db.by_col('NSA')

    Since we want to perform tests for spatial dependence, we need to specify
    the spatial weights matrix that includes the spatial configuration of the
    observations into the error component of the model. To do that, we can open
    an already existing gal file or create a new one. In this case, we will
    create one from ``columbus.shp``.

    >>> w = pysal.rook_from_shapefile(pysal.examples.get_path("columbus.shp"))

    Unless there is a good reason not to do it, the weights have to be
    row-standardized so every row of the matrix sums to one. Among other
    things, this allows to interpret the spatial lag of a variable as the
    average value of the neighboring observations. In PySAL, this can be
    easily performed in the following way:

    >>> w.transform = 'r'

    >>> tslsr = TSLS_Regimes(y, x, yd, q, regimes, w=w, constant_regi='many', spat_diag=False, name_y=y_var, name_x=x_var, name_yend=yd_var, name_q=q_var, name_ds='columbus', name_w='columbus.gal')

    >>> print tslsr.summary

    """
    def __init__(self, y, x, yend, q, regimes,\
             w=None, robust=None, gwk=None, sig2n_k=True,\
             spat_diag=False, vm=False, constant_regi='many',\
             cols2regi='all', name_y=None, name_x=None,\
             name_yend=None, name_q=None,\
             name_w=None, name_gwk=None, name_ds=None):

        n = USER.check_arrays(y, x)
        USER.check_y(y, n)
        USER.check_weights(w, y)
        USER.check_robust(robust, gwk)
        USER.check_spat_diag(spat_diag, w)
        x, name_x = REGI.Regimes_Frame.__init__(self, x, name_x, \
                regimes, constant_regi, cols2regi)
        yend, name_yend = REGI.Regimes_Frame.__init__(self, yend, \
                name_yend, regimes, constant_regi=None, cols2regi=cols2regi)
        q, name_q = REGI.Regimes_Frame.__init__(self, q, name_q, \
                regimes, constant_regi=None, cols2regi=cols2regi)
        BaseTSLS.__init__(self, y=y, x=x, yend=yend, q=q, \
                robust=robust, gwk=gwk, sig2n_k=sig2n_k)

        self.title = "TWO STAGE LEAST SQUARES - REGIMES"
        self.z = sphstack(x,yend)
        self.name_ds = USER.set_name_ds(name_ds)
        self.name_y = USER.set_name_y(name_y)
        self.name_x = USER.set_name_x(name_x, x, regi=True)
        self.name_yend = USER.set_name_yend(name_yend, yend)
        self.name_z = self.name_x + self.name_yend
        self.name_q = USER.set_name_q(name_q, q)
        #self.name_h = USER.set_name_h(self.name_x, self.name_q)
        self.robust = USER.set_robust(robust)
        self.name_w = USER.set_name_w(name_w, w)
        self.name_gwk = USER.set_name_w(name_gwk, gwk)
        """
        ##### DELETE BEFORE ADDING TO PYSAL: #####
        h = sphstack(x,q)
        h_r,_a_ = REGI.Regimes_Frame.__init__(self, h, None, \
                regimes, constant_regi, cols2regi)
        reg1 = BaseOLS(y=yend, x=h_r, robust=robust, gwk=gwk, \
                sig2n_k=sig2n_k)
        x2 = sphstack(x,reg1.predy)
        name_x.extend(name_yend)
        x2_r, name_x = REGI.Regimes_Frame.__init__(self, x2, name_x, \
                regimes, constant_regi, cols2regi)    
        BaseOLS.__init__(self, y=y, x=x2_r, robust=robust, gwk=gwk, \
                sig2n_k=sig2n_k)        
        self.title = "TWO STAGE LEAST SQUARES - REGIMES"
        #self.z = sphstack(x,yend)
        self.name_ds = USER.set_name_ds(name_ds)
        self.name_y = USER.set_name_y(name_y)
        self.name_x = USER.set_name_x(name_x, x, regi=True)
        self.name_yend = USER.set_name_yend(name_yend, yend)
        self.name_z = self.name_x + self.name_yend
        self.name_q = USER.set_name_q(name_q, q)
        self.name_h = USER.set_name_h(self.name_x, self.name_q)
        self.robust = USER.set_robust(robust)
        self.name_w = USER.set_name_w(name_w, w)
        self.name_gwk = USER.set_name_w(name_gwk, gwk)
        SUMMARY.OLS(reg=self, vm=vm, w=w, nonspat_diag=False,\
                    spat_diag=spat_diag, moran=False)
        """
        SUMMARY.TSLS(reg=self, vm=vm, w=w, spat_diag=spat_diag)


def _test():
    import doctest
    start_suppress = np.get_printoptions()['suppress']
    np.set_printoptions(suppress=True)    
    doctest.testmod()
    np.set_printoptions(suppress=start_suppress)


if __name__ == '__main__':
    #_test()        
    
    import numpy as np
    import pysal
    db = pysal.open(pysal.examples.get_path("columbus.dbf"),'r')
    y_var = 'CRIME'
    y = np.array([db.by_col(y_var)]).reshape(49,1)
    x_var = ['INC']
    x = np.array([db.by_col(name) for name in x_var]).T
    yd_var = ['HOVAL']
    yd = np.array([db.by_col(name) for name in yd_var]).T
    q_var = ['DISCBD']
    q = np.array([db.by_col(name) for name in q_var]).T
    regimes = db.by_col('NSA')
    w = pysal.rook_from_shapefile(pysal.examples.get_path("columbus.shp"))
    w.transform = 'r'
    tslsr = TSLS_Regimes(y, x, yd, q, regimes, w=w, constant_regi='many', spat_diag=False, name_y=y_var, name_x=x_var, name_yend=yd_var, name_q=q_var, name_ds='columbus', name_w='columbus.gal')
    print tslsr.summary

