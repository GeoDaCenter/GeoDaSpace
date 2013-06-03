"""
Spatial Seemingly Unrelated Regressions (SUR) estimation.
"""
__author__ = "Luc Anselin luc.anselin@asu.edu, Pedro V. Amaral pedro.amaral@asu.edu"

import numpy as np
import numpy.linalg as la
import multiprocessing as mp
import scipy.sparse as SP
import summary_output as SUMMARY
import user_output as USER
from utils import spdot, set_endog_sparse, sp_att, set_warn
from twosls import BaseTSLS
from platform import system

class GM_Endog_SUR():
    '''
    Parameters
    ----------
    y            : array
                   n*n_eqx1 array for dependent variable
    x            : array
                   Two dimensional array with n*n_eq rows and one column for each
                   independent (exogenous) variable, excluding the constant
    yend         : array
                   Two dimensional array with n*n_eq rows and one column for each
                   endogenous variable
    q            : array
                   Two dimensional array with n*n_eq rows and one column for each
                   external exogenous variable to use as instruments  
                   (note: this should not contain any variables from x);                   
    equationID   : list
                   List of n*n_eq values with the mapping of each
                   observation to a regime. Assumed to be aligned with 'x'.  
    sig2n_k      : boolean
                   If True, then use n-k to estimate sigma^2. If False, use n (default).
    cores        : integer
                   Number of cores to be used in the estimation (default: maximum available)
    vm           : boolean
                   If True, include variance-covariance matrix in summary
    name_y       : string
                   Name of dependent variable for use in output
    name_x       : list of strings
                   Names of independent variables for use in output
    name_yend    : list of strings
                   Names of endogenous variables for use in output
    name_q       : list of strings
                   Names of instruments for use in output
    name_equationID : list of strings
                     Names of equation indicator for use in output
    name_ds      : string
                   Name of dataset for use in output
    name_w       : string
                   Name of weights matrix for use in output


    Attributes
    ----------
    summary      : string
                   Summary of regression results and diagnostics (note: use in
                   conjunction with the print command)
    betas        : array
                   k*n_eqx1 array of estimated coefficients
    u            : array
                   n*n_eqx1 array of residuals
    predy        : array
                   n*n_eqx1 array of predicted y values
    n            : integer
                   Number of observations
    n_eq         : integer
                   Number of equations
    k            : integer
                   Number of variables for which coefficients are estimated
                   (including the constant)
    y            : array
                   n*n_eqx1 array for dependent variable
    x            : array
                   Two dimensional array with n*n_eq rows and one column for each
                   independent (exogenous) variable, including the constant
    yend         : array
                   Two dimensional array with n*n_eq rows and one column for each
                   endogenous variable
    q            : array
                   Two dimensional array with n*n_eq rows and one column for each
                   external exogenous variable used as instruments 
    z            : array
                   n*n_eqxk array of variables (combination of x and yend)
    h            : array
                   n*n_eqxl array of instruments (combination of x and q)
    vm           : array
                   Variance covariance matrix (k*n_eqxk*n_eq)
    std_err      : array
                   1xk*n_eq array of standard errors of the betas    
    z_stat       : list of tuples
                   z statistic; each tuple contains the pair (statistic,
                   p-value), where each is a float
    name_y        : string
                    Name of dependent variable for use in output
    name_x        : list of strings
                    Names of independent variables for use in output
    name_yend     : list of strings
                    Names of endogenous variables for use in output
    name_z        : list of strings
                    Names of exogenous and endogenous variables for use in 
                    output
    name_q        : list of strings
                    Names of external instruments
    name_h        : list of strings
                    Names of all instruments used in ouput
    name_ds       : string
                    Name of dataset for use in output
    name_equationID : list of strings
                     Names of equation indicator for use in output
    name_w       : string
                   Name of weights matrix for use in output
    title         : string
                    Name of the regression method used


    References
    ----------

    .. [1] Zellner, A., and H. Theil. 1962. Three stage least squares: Simultaneous
        estimate of simultaneous equations. Econometrica 29: 54-78

    Examples
    --------

    We first need to import the needed modules, namely numpy to convert the
    data we read into arrays that ``spreg`` understands and ``pysal`` to
    perform all the analysis.

    >>> import numpy as np
    >>> import pysal

    Open data on NCOVR US County Homicides (3085 areas) using pysal.open().
    This is the DBF associated with the NAT shapefile.  Note that
    pysal.open() also reads data in CSV format; since the actual class
    requires data to be passed in as numpy arrays, the user can read their
    data in using any method.  

    >>> db = pysal.open(pysal.examples.get_path("NAT.dbf"),'r')
 
    The SUR models assume that data is provided in 'long' format, i.e.
    data for all equations is stacked vertically, with a column working
    as the equation identifier. So in this example, we will first prepare our
    data in the 'long' format by stacking the variables for years 1980 and 1990.
    Extract the HR columns (homicide rates) from the DBF file and make it the
    dependent variable for the regression.

    >>> y_var = ['HR80','HR90']
    >>> y = np.vstack((np.array([db.by_col(r)]).T for r in y_var))
    >>> y_name = 'HR'
    
    Extract UE (unemployment rate) and PS (population structure) vectors from
    the DBF to be used as independent variables in the regression. Other variables
    can be inserted by adding their names to x_var, such as x_var = ['Var1','Var2','...]
    Note that PySAL requires this to be an nxj numpy array, where j is the
    number of independent variables (not including a constant). By default
    this model adds a vector of ones to the independent variables passed in.

    >>> x_var = ['PS80','UE80','PS90','UE90']
    >>> x = np.array([db.by_col(name) for name in x_var]).T
    >>> x = np.vstack((x[:,0:2],x[:,2:4]))
    >>> x_name = ['PS','UE']

    In this case we consider RD (resource deprivation) as an endogenous regressor.
    We tell the model that this is so by passing it in a different parameter
    from the exogenous variables (x).

    >>> yd_var = ['RD80','RD90']
    >>> yd = np.vstack((np.array([db.by_col(r)]).T for r in yd_var))
    >>> yd_name = ['RD']

    Because we have endogenous variables, to obtain a correct estimate of the
    model, we need to instrument for RD. We use FP (families below poverty)
    for this and hence put it in the instruments parameter, 'q'.

    >>> q_var = ['FP79','FP89']
    >>> q = np.vstack((np.array([db.by_col(r)]).T for r in q_var))
    >>> q_name = ['FP']

    The different equations in this data are given according to the time period.
    Since we stacked the data, we know the order of the time periods:

    >>> eq_name = 'Year'
    >>> eq_ID = [1980]*3085 + [1990]*3085

    We can now run the regression and then have a summary of the output
    by typing: model.summary
    Alternatively, we can just check the betas and standard errors of the
    parameters:

    >>> three_sls = GM_Endog_SUR(y, x, yd, q, eq_ID, name_y=y_name, name_x=x_name, name_yend=yd_name, name_q=q_name, name_equationID=eq_name, name_ds='NAT')

    >>> three_sls.betas
    array([[ 6.92426353],
           [ 1.42921826],
           [ 0.00049435],
           [ 3.5829275 ],
           [ 7.62385875],
           [ 1.65031181],
           [-0.21682974],
           [ 3.91250428]])

    >>> three_sls.std_err
    array([ 0.23220853,  0.10373417,  0.03086193,  0.11131999,  0.28739415,
            0.09597031,  0.04089547,  0.13586789])
    '''    
    
    def __init__(self, y, x, yend, q, equationID, w=None,\
                 cores=None, sig2n_k=False, name_y=None, name_x=None, name_yend=None,\
                 name_q=None, name_equationID=None, name_ds=None, name_w=None, vm=False,\
                 w_lags=None,lag_q=None):

        self.equationID = equationID
        eq_set = list(set(self.equationID))
        self.eq_set = eq_set
        self.n_eq = len(eq_set)
        if w:
            self.n = w.n            
            assert self.n_eq*w.n==y.shape[0], "Number of equations, weights dimension and lenght of vector Y are not aligned."
            ws = w.sparse
        else:
            if w_lags:
                raise Exception, "W matrix required to run spatial lag model."
            ws = None
        eq_ids = dict((r, list(np.where(np.array(equationID) == r)[0])) for r in eq_set)        

        #Running 2SLS for each equation separately
        stp2 = {}
        if system() == 'Windows':
            for r in eq_set:
                stp2[r] = _run_stp1(y,x,yend,q,eq_ids,r,sig2n_k,ws,w_lags,lag_q)
            results_stp2 = stp2
        else:
            pool = mp.Pool(cores)
            for r in eq_set:
                stp2[r] = pool.apply_async(_run_stp1,args=(y,x,yend,q,eq_ids,r,sig2n_k,ws,w_lags,lag_q, ))
            pool.close()
            pool.join()
            results_stp2 = dict((r, stp2[r].get()) for r in eq_set)

        if not w:
            self.n = results_stp2[eq_set[0]].n
            assert self.n_eq*self.n==y.shape[0], "Number of equations and lenght of vector Y are not aligned."

        #Building sigma matrix
        if sig2n_k:
            dof = list(results_stp2[r].n - results_stp2[r].k for r in eq_set)
            dof = np.array(dof).reshape(self.n_eq,1)
            den=np.dot(dof,dof.T)**0.5
        else:
            den = np.array([float(self.n)]*self.n_eq**2).reshape(self.n_eq,self.n_eq)
        BigU1 = np.hstack((results_stp2[r].u for r in eq_set))
        sig = la.inv(np.dot(BigU1.T,BigU1)/den)

        #Building stacked matrices:
        BigZ = np.hstack((results_stp2[r].z.flatten() for r in eq_set))
        BigZhat = np.hstack((results_stp2[r].zhat.flatten() for r in eq_set))
        k_z = 0
        indices_z,indptr_z = [],[]
        for r in eq_set:
            indices_z += range(k_z,results_stp2[r].z.shape[1]+k_z)*self.n
            indptr_z += list(np.arange(0,self.n)*results_stp2[r].z.shape[1] + k_z*self.n)
            k_z += results_stp2[r].z.shape[1]
        BigZ = SP.csr_matrix((BigZ,indices_z,indptr_z+[BigZ.shape[0]]))            
        BigZhat = SP.csr_matrix((BigZhat,indices_z,indptr_z+[BigZhat.shape[0]]))
        BigY = np.vstack((results_stp2[r].y for r in eq_set))

        #Estimating parameters
        BigZhattsig = np.vstack((np.hstack((results_stp2[eq_set[i]].zhat.T*sig[i,j] for j in range(self.n_eq))) for i in range(self.n_eq)))
        #BigZhattsig = BigZhat.T*SP.kron(sig,SP.identity(self.n))
        self.vm = la.inv(spdot(BigZhattsig,BigZhat))
        fact2 = spdot(BigZhattsig,BigY)
        self.betas = spdot(self.vm,fact2)
        self.std_err = np.sqrt(self.vm.diagonal())

        #Prepare output
        self.multi = results_stp2
        k_b = 0
        for r in eq_set:
            k_b1 = self.multi[r].betas.shape[0]
            self.multi[r].betas[:,] = self.betas[k_b:k_b+k_b1,]
            self.multi[r].vm = self.vm[k_b:k_b+k_b1,k_b:k_b+k_b1]
            k_b += k_b1
            self.multi[r].predy = spdot(self.multi[r].z,self.multi[r].betas)
            self.multi[r].u = self.multi[r].y - self.multi[r].predy

        if not w_lags:
            title = "THREE-STAGE LEAST SQUARES - EQUATION "            
            self.multi = USER.set_name_multi(self.multi,eq_set,name_equationID,y,x,name_y,name_x,name_ds,title,name_w,robust=None,endog=(yend,q,name_yend,name_q),sp_lag=False)
            SUMMARY.TSLS_multi(reg=self, multireg=self.multi, vm=vm, spat_diag=False, sur=True)



class GM_Lag_SUR(GM_Endog_SUR):
    '''
    Parameters
    ----------
    y            : array
                   n*n_eqx1 array for dependent variable
    x            : array
                   Two dimensional array with n*n_eq rows and one column for each
                   independent (exogenous) variable, excluding the constant
    yend         : array
                   Two dimensional array with n*n_eq rows and one column for each
                   endogenous variable
    q            : array
                   Two dimensional array with n*n_eq rows and one column for each
                   external exogenous variable to use as instruments (note: 
                   this should not contain any variables from x);                   
    equationID   : list
                   List of n*n_eq values with the mapping of each
                   observation to a regime. Assumed to be aligned with 'x'.  
    w            : pysal W object
                   Spatial weights object
    sig2n_k      : boolean
                   If True, then use n-k to estimate sigma^2. If False, use n (default).
    cores        : integer
                   Number of cores to be used in the estimation (default: maximum available)
    vm           : boolean
                   If True, include variance-covariance matrix in summary
    name_y       : string
                   Name of dependent variable for use in output
    name_x       : list of strings
                   Names of independent variables for use in output
    name_yend    : list of strings
                   Names of endogenous variables for use in output
    name_q       : list of strings
                   Names of instruments for use in output
    name_equationID : list of strings
                     Names of equation indicator for use in output
    name_ds      : string
                   Name of dataset for use in output
    name_w       : string
                   Name of weights matrix for use in output


    Attributes
    ----------
    summary      : string
                   Summary of regression results and diagnostics (note: use in
                   conjunction with the print command)
    betas        : array
                   k*n_eqx1 array of estimated coefficients
    u            : array
                   n*n_eqx1 array of residuals
    predy        : array
                   n*n_eqx1 array of predicted y values
    n            : integer
                   Number of observations
    n_eq         : integer
                   Number of equations
    k            : integer
                   Number of variables for which coefficients are estimated
                   (including the constant)
    y            : array
                   n*n_eqx1 array for dependent variable
    x            : array
                   Two dimensional array with n*n_eq rows and one column for each
                   independent (exogenous) variable, including the constant
    yend         : array
                   Two dimensional array with n*n_eq rows and one column for each
                   endogenous variable
    q            : array
                   Two dimensional array with n*n_eq rows and one column for each
                   external exogenous variable used as instruments 
    z            : array
                   n*n_eqxk array of variables (combination of x and yend)
    h            : array
                   n*n_eqxl array of instruments (combination of x and q)
    vm           : array
                   Variance covariance matrix (k*n_eqxk*n_eq)
    std_err      : array
                   1xk*n_eq array of standard errors of the betas    
    z_stat       : list of tuples
                   z statistic; each tuple contains the pair (statistic,
                   p-value), where each is a float
    name_y        : string
                    Name of dependent variable for use in output
    name_x        : list of strings
                    Names of independent variables for use in output
    name_yend     : list of strings
                    Names of endogenous variables for use in output
    name_z        : list of strings
                    Names of exogenous and endogenous variables for use in 
                    output
    name_q        : list of strings
                    Names of external instruments
    name_h        : list of strings
                    Names of all instruments used in ouput
    name_ds       : string
                    Name of dataset for use in output
    name_equationID : list of strings
                     Names of equation indicator for use in output
    name_w       : string
                   Name of weights matrix for use in output
    title         : string
                    Name of the regression method used


    References
    ----------

    .. [1] Zellner, A., and H. Theil. 1962. Three stage least squares: Simultaneous
        estimate of simultaneous equations. Econometrica 29: 54-78

    Examples
    --------

    We first need to import the needed modules, namely numpy to convert the
    data we read into arrays that ``spreg`` understands and ``pysal`` to
    perform all the analysis.

    >>> import numpy as np
    >>> import pysal

    Open data on NCOVR US County Homicides (3085 areas) using pysal.open().
    This is the DBF associated with the NAT shapefile.  Note that
    pysal.open() also reads data in CSV format; since the actual class
    requires data to be passed in as numpy arrays, the user can read their
    data in using any method.  

    >>> db = pysal.open(pysal.examples.get_path("NAT.dbf"),'r')
 
    The SUR models assume that data is provided in 'long' format, i.e.
    data for all equations is stacked vertically, with a column working
    as the equation identifier. So in this example, we will first prepare our
    data in the 'long' format by stacking the variables for years 1980 and 1990.
    Extract the HR columns (homicide rates) from the DBF file and make it the
    dependent variable for the regression.

    >>> y_var = ['HR80','HR90']
    >>> y = np.vstack((np.array([db.by_col(r)]).T for r in y_var))
    >>> y_name = 'HR'
    
    Extract UE (unemployment rate) and PS (population structure) vectors from
    the DBF to be used as independent variables in the regression. Other variables
    can be inserted by adding their names to x_var, such as x_var = ['Var1','Var2','...]
    Note that PySAL requires this to be an nxj numpy array, where j is the
    number of independent variables (not including a constant). By default
    this model adds a vector of ones to the independent variables passed in.

    >>> x_var = ['PS80','UE80','PS90','UE90']
    >>> x = np.array([db.by_col(name) for name in x_var]).T
    >>> x = np.vstack((x[:,0:2],x[:,2:4]))
    >>> x_name = ['PS','UE']

    In this case we consider RD (resource deprivation) as an endogenous regressor.
    We tell the model that this is so by passing it in a different parameter
    from the exogenous variables (x).

    >>> yd_var = ['RD80','RD90']
    >>> yd = np.vstack((np.array([db.by_col(r)]).T for r in yd_var))
    >>> yd_name = ['RD']

    Because we have endogenous variables, to obtain a correct estimate of the
    model, we need to instrument for RD. We use FP (families below poverty)
    for this and hence put it in the instruments parameter, 'q'.

    >>> q_var = ['FP79','FP89']
    >>> q = np.vstack((np.array([db.by_col(r)]).T for r in q_var))
    >>> q_name = ['FP']

    The different equations in this data are given according to the time period.
    Since we stacked the data, we know the order of the time periods:

    >>> eq_name = 'Year'
    >>> eq_ID = [1980]*3085 + [1990]*3085

    Since we want to estimate a spatial lag model, we need to specify
    the spatial weights matrix that includes the spatial configuration of the
    observations. To do that, we can open an already existing gal file or create 
    a new one. In this case, we will create one from ``NAT.shp``.

    >>> w = pysal.rook_from_shapefile(pysal.examples.get_path("NAT.shp"))

    Unless there is a good reason not to do it, the weights have to be
    row-standardized so every row of the matrix sums to one. Among other
    things, this allows to interpret the spatial lag of a variable as the
    average value of the neighboring observations. In PySAL, this can be
    easily performed in the following way:

    >>> w.transform = 'r'

    We can now run the regression and then have a summary of the output
    by typing: model.summary
    Alternatively, we can just check the betas and standard errors of the
    parameters:

    >>> three_sls = GM_Lag_SUR(y, x, yd, q, eq_ID, w, name_y=y_name, name_x=x_name, name_yend=yd_name, name_q=q_name, name_equationID=eq_name, name_ds='NAT', name_w='NAT.shp')

    >>> three_sls.betas
    array([[ 6.90380817],
           [ 1.43423452],
           [-0.00660443],
           [ 3.63050212],
           [ 0.00987183],
           [ 5.73006253],
           [ 1.39970248],
           [-0.16020237],
           [ 3.23386107],
           [ 0.24463027]])

    >>> three_sls.std_err
    array([ 0.48781006,  0.11535462,  0.03188754,  0.18749647,  0.05442233,
            0.44911913,  0.10469721,  0.04147879,  0.19080703,  0.04405396])
    '''    
    
    def __init__(self, y, x, yend, q, equationID, w,\
                 cores=None, sig2n_k=False, name_y=None, name_x=None, name_yend=None,\
                 name_q=None, name_equationID=None, name_ds=None, name_w=None, vm=False,\
                 w_lags=1, lag_q=True):

        GM_Endog_SUR.__init__(self, y=y, x=x, yend=yend, q=q, equationID=equationID, w=w, cores=cores,\
                 sig2n_k=sig2n_k, name_y=name_y, name_x=name_x, name_yend=name_yend, name_q=name_q,\
                 name_equationID=name_equationID, name_ds=name_ds, name_w=name_w, vm=vm,\
                 w_lags=w_lags, lag_q=lag_q)

        for r in self.eq_set:
            self.multi[r].predy_e, self.multi[r].e_pred, warn = sp_att(w,self.multi[r].y,self.multi[r].predy,\
                          self.multi[r].yend[:,-1].reshape(self.n,1),self.multi[r].betas[-1])
            set_warn(self.multi[r], warn)
        
        title = "SPATIAL THREE-STAGE LEAST SQUARES - EQUATION "
        self.multi = USER.set_name_multi(self.multi,self.eq_set,name_equationID,y,x,name_y,name_x,name_ds,title,name_w,robust=None,endog=(yend,q,name_yend,name_q),sp_lag=(w_lags, lag_q))
        SUMMARY.GM_Lag_multi(reg=self, multireg=self.multi, vm=vm, spat_diag=False, sur=True)

def _run_stp1(y,x,yend,q,eq_ids,r,sig2n_k,w,w_lags,lag_q):
    y_r = y[eq_ids[r]]
    x_r = x[eq_ids[r]]
    yend_r = yend[eq_ids[r]]
    q_r = q[eq_ids[r]]
    if w_lags:
        yend_r, q_r = set_endog_sparse(y_r, x_r, w, yend_r, q_r, w_lags, lag_q)        
    x_constant = USER.check_constant(x_r)
    model = BaseTSLS(y_r, x_constant, yend_r, q_r, sig2n_k=sig2n_k) 
    model.zhat = np.dot(model.h,np.dot(model.hthi,model.htz))
    return model
        
       
def _test():
    import doctest
    start_suppress = np.get_printoptions()['suppress']
    np.set_printoptions(suppress=True)    
    doctest.testmod()
    np.set_printoptions(suppress=start_suppress)


if __name__ == '__main__':
    _test()          

    import pysal
    import numpy as np
    """
    db = pysal.open('examples/swohio.csv','r')
    y_var = 'wage'
    y = np.array([db.by_col(y_var)]).T
    x_var = ['smsa']
    x = np.array([db.by_col(name) for name in x_var]).T
    yend_var = ['UN']
    yend = np.array([db.by_col(name) for name in yend_var]).T
    q_var = ['UN']
    q = np.array([db.by_col(name) for name in q_var]).T
    eq_var = 'time'
    equationID = db.by_col(eq_var)
    w = pysal.open('examples/swohio.gal','r').read()
    w.transform = 'r'

    sur1=GM_Endog_SUR(y, x, yend, q, equationID)
    print sur1.summary
    """
    db = pysal.open(pysal.examples.get_path("NAT.dbf"),'r')
    y_var = ['HR80','HR90']
    y = np.vstack((np.array([db.by_col(r)]).T for r in y_var))
    y_name = 'HR'
    x_var = ['PS80','UE80','PS90','UE90']
    x = np.array([db.by_col(name) for name in x_var]).T
    x = np.vstack((x[:,0:2],x[:,2:4]))
    x_name = ['PS','UE']
    yd_var = ['RD80','RD90']
    yd = np.vstack((np.array([db.by_col(r)]).T for r in yd_var))
    yd_name = ['RD']
    q_var = ['FP79','FP89']
    q = np.vstack((np.array([db.by_col(r)]).T for r in q_var))
    q_name = ['FP']
    eq_name = 'Year'
    eq_ID = [1980]*3085 + [1990]*3085
    w = pysal.rook_from_shapefile(pysal.examples.get_path("NAT.shp"))
    w.transform = 'r'
    three_sls = GM_Lag_SUR(y, x, yd, q, eq_ID, w=w, name_y=y_name, name_x=x_name, name_yend=yd_name, name_q=q_name, name_equationID=eq_name, name_ds='NAT')
    print three_sls.summary
    #"""
