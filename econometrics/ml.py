"""
Maximum likelihood estimation of spatial process models


TODO
----

 - asymptotic standard errors for ml_lag do no exactly match
   opengeoda
 - ml_error vcv
 - ml_error likelihood
 - document
 - write tests
 - wrapper classes
 - integration with spreg

"""

__author__ = "Serge Rey srey@asu.edu, Luc Anselin luc.anselin@asu.edu"

import numpy as np
import pysal as ps
import scipy.sparse as SPARSE


def defl_lag(r, w, e1, e2):
    """
    Derivative of the likelihood for the lag model

    Parameters
    ----------

    r: estimate of rho

    w: spatial weights object

    e1: ols residuals of y on X 

    e2: ols residuals of Wy on X

    Returns
    -------

    tuple: 
        tuple[0]: dfl (scalar) value of derivative at parameter values
        tuple[1]: tr (scalar) trace of  (I-rW)^{-1} W

    """
    n = w.n
    e1re2 = e1 - r * e2
    num = np.dot(e1re2.T, e2)
    den = np.dot(e1re2.T, e1re2)
    a = -r * w.full()[0]
    np.fill_diagonal(a, np.ones((w.n,1)))
    a = np.linalg.inv(a)
    tr = np.trace(np.dot(a, w.full()[0]))
    dfl = n * (num/den) - tr
    return dfl[0][0], tr

def log_like_lag_full(w,b,X,y):
    r = b[0]
    ldet = _logJacobian(w, r)
    return log_like_lag(ldet, w, b, X, y)

def log_like_lag_ord(w, b, X, y, evals):
    r = b[0]
    revals = r * evals
    ldet = np.log(1-revals).sum()
    return log_like_lag(ldet, w, b, X, y)

def log_like_lag(ldet, w, b, X, y):
    n = w.n
    r = b[0]    # ml estimate of rho
    b = b[1:]   # ml for betas
    yl = ps.lag_spatial(w,y)
    ys = y - r * yl
    XX = np.dot(X.T, X)
    iXX = np.linalg.inv(XX)
    b = np.dot(iXX, np.dot(X.T,ys))
    yhat = r * yl + np.dot(X,b)
    e = y - yhat
    e2 = (e**2).sum()
    sig2 = e2 / n
    ln2pi = np.log(2*np.pi)
    return ldet - n/2. * ln2pi - n/2. * np.log(sig2) - e2/(2 * sig2)

def log_lik_error(ldet, w, b, lam, X, y):
    n = w.n
    yl = ps.lag_spatial(w,y)
    ys = y - lam * yl
    XS = X - lam * ps.lag_spatial(w, X)
    iXSXS = np.linalg.inv(np.dot(XS.T, XS))
    es = y  - ys - np.dot(X,b) + np.dot(XS,b)
    es2 = (es**2).sum()
    sig2 = es2 / n
    ln2pi = np.log(2*np.pi)
    return  ldet - n/2. * ln2pi - n/2. * np.log(sig2) - es2 / (2 * sig2)

def _logJacobian(w, rho):
    """
    Log determinant of I-\rho W

    Brute force 
    """

    return np.log(np.linalg.det(np.eye(w.n) - rho * w.full()[0]))

def symmetrize(w):
    """Generate symmetric matrix that has same eigenvalues as an asymmetric row
    standardized matrix w

    Parameters
    ----------
    w: weights object that has been row standardized

    Returns
    -------
    a sparse symmetric matrix with same eigenvalues as w
    
    """
    current = w.transform
    w.transform = 'B'
    d = w.sparse.sum(axis=1) # row sum
    d.shape=(w.n,)
    D12 = SPARSE.spdiags(np.sqrt(1./d),[0],w.n,w.n)
    w.transform = current
    return D12*w.sparse*D12

def defer(w, lam, yy, yyl, ylyl, Xy, Xly, Xlyl, XX, XlX, XlXl):
    """
    Partial derivative of concentrated log-likelihood for error model.


    Parameters
    ----------

    w: spatial weights object

    lam: estimate of autoregressive coefficient

    yy: sum of squares of y

    yyl: y'Wy

    ylyl: (Wy)'Wy

    Xy: X'y

    Xly:  (WX)'y


    Xlyl:  (WX)'Wy

    XlX: (WX)' X

    XLXL: (WX)'WX
    """

    n = w.n

    dlik = 0
    flag = 0
    lam2 = lam**2
    tlam = 2.0 * lam

    # weighted ls for lam
    dd = np.linalg.inv(XX - lam * XlX + lam2 * XlXl)
    longyX = Xy - lam * Xly + lam2 * Xlyl
    b = np.dot(dd, longyX)
    shortyX = -Xly + tlam * Xlyl

    doubX = np.dot(b.T, np.dot(XX,b)) - lam * np.dot(b.T, np.dot(XlX,b))
    doubX = doubX + lam2 * np.dot(b.T, np.dot(XlXl, b))
    ee = yy - tlam * yyl + lam2 *ylyl - 2.0 * np.dot(longyX.T, b) + doubX
    sig2 = ee/n

    # trace
    a = -lam * w.full()[0]
    np.fill_diagonal(a, np.ones((w.n,1)))
    a = np.linalg.inv(a)
    tr = np.trace(np.dot(a, w.full()[0]))

    dlik = -2.0 * yyl + tlam * ylyl
    dlik = dlik - 2.0 * np.dot(shortyX.T, b)
    dlik = dlik + np.dot(b.T, np.dot( (-XlX + tlam * XlXl), b))
    dlik = -0.5 * dlik / sig2 - tr
    return (dlik[0][0], b, sig2, tr, dd)

def ml_error(y, X, w, precrit=0.0000001, verbose=False, like='full'):
    """
    Maximum likelihood of spatial error model

    Parameters
    ----------
    y: dependent variable (nx1 array)

    w: spatial weights object

    X: explanatory variables (nxk array)

    precrit: convergence criterion

    verbose: boolen to print iterations in estimation

    like: method to use for evaluating concentrated likelihood function
    (FULL|ORD) where FULL=Brute Force, ORD = eigenvalue based jacobian

    Returns
    -------

    Results: dictionary with estimates, standard errors, vcv, and z-values

    """
    n = w.n
    n,k = X.shape
    yy = (y**2).sum()
    yl = ps.lag_spatial(w, y)
    ylyl = (yl**2).sum()
    Xy = np.dot(X.T,y)
    Xl = ps.lag_spatial(w, X)
    Xly = np.dot(Xl.T,y) + np.dot(X.T, yl)
    Xlyl = np.dot(Xl.T, yl)
    XX = np.dot(X.T, X)
    XlX = np.dot(Xl.T,X) + np.dot(X.T, Xl)
    XlXl = np.dot(Xl.T, Xl)
    yly = np.dot(yl.T, y)
    yyl = np.dot(y.T, yl)
    ylyl = np.dot(yl.T, yl)


    lam = 0
    dlik, b, sig2, tr, dd = defer(w, lam, yy, yyl, ylyl, Xy, Xly, Xlyl, XX, XlX,
            XlXl)

    roots = np.linalg.eigvals(w.full()[0])
    maxroot = 1./roots.max()
    minroot = 1./roots.min()
    delta = 0.0001



    if dlik > 0:
        ll = lam
        ul = maxroot - delta
    else:
        ul = lam
        ll = minroot + delta

    # bisection
    t = 10

    lam0 = (ll + ul) /2.
    i = 0

    if verbose:
        line ="\nMaximum Likelihood Estimation of Spatial error Model"
        print line
        line ="%-5s\t%12s\t%12s\t%12s\t%12s"%("Iter.","LL","LAMBDA","UL","dlik")
        print line

    while abs(t - lam0)  > precrit:
        if verbose:
            print "%d\t%12.8f\t%12.8f\t%12.8f\t%12.8f"  % (i,ll, lam0, ul, dlik)
        i += 1

        dlik, b, sig2, tr, dd = defer(w, lam0, yy, yyl, ylyl, Xy, Xly, Xlyl,
                XX, XlX, XlXl)
        if dlik > 0:
            ll = lam0
        else:
            ul = lam0
        t = lam0
        lam0 = (ul + ll)/ 2.

    #if like.upper() == 'ORD':
    #    evals = SPARSE.linalg.eighsh(symmetrize(w))[0]
    #    llik = log_lik_error_ord(w, b, X, y, evals)
    #elif like.upper() == 'FULL':
    #    llik = log_like_error_full(w, b, X, y)

    ldet = _logJacobian(w, lam0)
    llik = log_lik_error(ldet, w, b, lam0, X, y)



    results = {}
    results['betas'] = b
    results['lambda'] = lam0
    results['llik'] = llik
    return results

def ml_lag(y, X, w,  precrit=0.0000001, verbose=False, like='full'):
    """
    Maximum likelihood estimation of spatial lag model

    Parameters
    ----------

    y: dependent variable (nx1 array)

    w: spatial weights object

    X: explanatory variables (nxk array)

    precrit: convergence criterion

    verbose: boolen to print iterations in estimation

    like: method to use for evaluating concentrated likelihood function
    (FULL|ORD) where FULL=Brute Force, ORD = eigenvalue based jacobian

    Returns
    -------

    Results: dictionary with estimates, standard errors, vcv, and z-values
    """

    # step 1 OLS of X on y yields b1
    d = np.linalg.inv(np.dot(X.T, X))
    b1 = np.dot(d, np.dot(X.T, y))

    # step 2 OLS of X on Wy: yields b2
    wy = ps.lag_spatial(w,y)
    b2 = np.dot(d, np.dot(X.T, wy))

    # step 3 compute residuals e1, e2
    e1 = y - np.dot(X,b1)
    e2 = wy - np.dot(X,b2)

    # step 4 given e1, e2 find rho that maximizes Lc

    # ols estimate of rho
    XA = np.hstack((wy,X))
    bols = np.dot(np.linalg.inv(np.dot(XA.T, XA)), np.dot(XA.T,y))
    rols = bols[0][0]

    while np.abs(rols) > 1.0:
        rols = rols/2.0

    if rols > 0.0:
        r1 = rols
        r2 = r1 / 5.0
    else:
        r2 = rols
        r1 = r2 / 5.0

    df1 = 0
    df2 = 0
    tr = 0
    df1, tr = defl_lag(r1, w, e1, e2)
    df2, tr = defl_lag(r2, w, e1, e2)

    if df1*df2 <= 0:
        ll = r2
        ul = r1
    elif df1 >= 0.0 and df1 >= df2:
        ll = -0.999
        ul = r2
        df1 = df2
        df2 = -(10.0**10)
    elif df1 >= 0.0 and df1 < df2:
        ll = r1
        ul = 0.999
        df2 = df1
        df1 = -(10.0**10)
    elif df1 < 0.0 and df1 >= df2:
        ul = 0.999
        ll = r1
        df2 = df1
        df1 = 10.0**10
    else:
        ul = r2
        ll = -0.999
        df1 = df2
        df2 = 10.0**10

    # main bisection iteration

    err = 10
    t = rols
    ro = (ll+ul) / 2.
    if verbose:
        line ="\nMaximum Likelihood Estimation of Spatial lag Model"
        print line
        line ="%-5s\t%12s\t%12s\t%12s\t%12s"%("Iter.","LL","RHO","UL","DFR")
        print line

    i = 0
    while err > precrit:
        if verbose:
            print "%d\t%12.8f\t%12.8f\t%12.8f\t%12.8f"  % (i,ll, ro, ul, df1)
        dfr, tr = defl_lag(ro, w, e1, e2)
        if dfr*df1 < 0.0:
            ll = ro
        else:
            ul = ro
            df1 = dfr
        err = np.abs(t-ro)
        t = ro
        ro =(ul+ll)/2.
        i += 1
    ro = t
    tr1 = tr
    bml = b1 - (ro * b2)
    b = [ro,bml]

    xb = np.dot(X, bml)
    eml = y - ro * wy - xb
    sig2 = (eml**2).sum() / w.n
    

    # Likelihood evaluation

    if like.upper() == 'ORD':
        evals = SPARSE.linalg.eigsh(symmetrize(w))[0]
        llik = log_like_lag_ord(w, b, X, y, evals)
    elif like.upper() == 'FULL':
        llik = log_like_lag_full(w, b, X, y)


    # Information matrix 
    # Ipp IpB  Ips
    # IBp IBB  IBs
    # Isp IsB  Iss

    # Ipp
    n,k = X.shape
    W = w.full()[0]
    A_inv = np.linalg.inv(np.eye(w.n) - ro * W)
    WA = np.dot(W, A_inv)
    tr1 = np.trace(WA)
    tr1 = tr1**2
    tr2 = np.trace(np.dot(WA.T, WA))
    WAXB = np.dot(WA,xb)
    Ipp = tr1 + tr2 + np.dot(WAXB.T, WAXB)/sig2

    I = np.eye(w.n)
    IpB = np.dot(X.T, WAXB).T / sig2
    Ips = np.trace(WA) / sig2

    IBp = IpB.T
    IBB = np.dot(X.T,X) / sig2

    Isp = Ips
    Iss = n / (2 * sig2 * sig2)

    results = {}
    results['betas'] = bml
    results['rho'] = ro
    results['llik'] = llik
    results['sig2'] = sig2
    dim = k + 2 
    Info = np.zeros((dim,dim))
    Info[0,0] = Ipp
    Info[0,1:k+1] = IpB
    Info[0,k+1] = Ips
    Info[1:k+1,0 ] = IpB
    Info[1:k+1, 1:k+1] = IBB
    Info[k+1,0 ] = Ips
    Info[k+1, k+1] = Iss
    VCV = np.linalg.inv(Info)
    se_b = np.sqrt(np.diag(VCV)[1:k+1])
    se_b.shape = (k,1)
    z_b = bml/se_b
    se_rho = np.sqrt(VCV[0,0])
    z_rho = ro / se_rho
    se_sig2 = np.sqrt(VCV[k+1,k+1])
    results['se_b'] = se_b
    results['z_b'] = z_b
    results['se_rho'] = se_rho
    results['z_rho'] = z_rho
    results['se_sig2'] = se_sig2
    results['VCV'] = VCV
    return results

if __name__ == '__main__':

    db =  ps.open(ps.examples.get_path('columbus.dbf'),'r')
    y = np.array(db.by_col("CRIME"))
    y.shape = (len(y),1)
    X = []
    X.append(db.by_col("INC"))
    X.append(db.by_col("HOVAL"))
    X = np.array(X).T
    #w = ps.open(ps.examples.get_path("columbus.gal")).read()
    w = ps.rook_from_shapefile(ps.examples.get_path("columbus.shp"))
    w.transform = 'r'
    X = np.hstack((np.ones((w.n,1)),X))
    bml = ml_lag(y,X,w, verbose=True, like='ORD')
    bmlf = ml_lag(y,X,w, verbose=True, like='FULL')
    bmle = ml_error(y,X,w, verbose=True)
