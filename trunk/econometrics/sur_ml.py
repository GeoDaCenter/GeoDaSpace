"""
Maximum Likelihood Estimation of Spatial Seemingly Unrelated Regressions (SUR).
"""
__author__ = "Luc Anselin luc.anselin@asu.edu, Pedro V. Amaral pedro.amaral@asu.edu"

import numpy as np
import numpy.linalg as la
import multiprocessing as mp
import scipy.sparse as SP
import summary_output as SUMMARY
import user_output as USER
from pysal import lag_spatial
from utils import spdot
from ols import BaseOLS
from platform import system



class _sur_frame():
    def __init__(self, y, x, equationID, w, cores):

        self.equationID = equationID
        eq_set = list(set(self.equationID))
        self.eq_set = eq_set
        self.n_eq = len(eq_set)
        self.k = x.shape[1]+1
        if w:
            self.n = w.n            
            assert self.n_eq*w.n==y.shape[0], "Number of equations, weights dimension and lenght of vector Y are not aligned."
            #ws = w.sparse
        #else:
        #    ws = None
        eq_ids = dict((r, list(np.where(np.array(equationID) == r)[0])) for r in eq_set)
        #Running OLS for each equation separately
        stp2 = {}
        if system() == 'Windows':
            for r in eq_set:
                stp2[r] = _run_stp1(y,x,eq_ids,r)
            results_stp2 = stp2
        else:
            pool = mp.Pool(cores)
            for r in eq_set:
                stp2[r] = pool.apply_async(_run_stp1,args=(y,x,eq_ids,r, ))
            pool.close()
            pool.join()
            results_stp2 = dict((r, stp2[r].get()) for r in eq_set)
            
        if not w:
            self.n = results_stp2[eq_set[0]].n
            assert self.n_eq*self.n==y.shape[0], "Number of equations and lenght of vector Y are not aligned."

        return  eq_ids, results_stp2


class ML_SUR(_sur_frame):
    def __init__(self, y, x, equationID, w=None,\
                 cores=None, name_y=None, name_x=None,\
                 name_equationID=None, name_ds=None, name_w=None, vm=False,\
                 maxiter=1000, epsilon=0.00001):
        
        eq_ids, results_stp2 = _sur_frame.__init__(self, y, x, equationID, w, cores)

        #Building stacked matrices:
        BigU1 = np.hstack((results_stp2[r].u for r in self.eq_set))
        BigY = np.hstack((results_stp2[r].y for r in self.eq_set))
        
        #Running SUR:
        sig1 = np.dot(BigU1.T,BigU1)
        try:        
            sig1inv = la.inv(sig1)
            sig1 = sig1/self.n
            sig1inv = sig1inv*self.n
            det = la.det(sig1inv)
        except:
            raise Exception, "ERROR: singular error variance matrix"

        xomx = np.zeros((self.n_eq*self.k,self.n_eq*self.k),float)
        xomy = np.zeros((self.n_eq*self.k,self.n_eq),float)

        k_temp = 0
        for r in self.eq_set:
            x_r1 = results_stp2[r].x
            y_r1_2 = np.hstack((np.dot(x_r1.T,results_stp2[j].y) for j in self.eq_set))
            x_r1_2 = np.hstack((np.dot(x_r1.T,results_stp2[j].x) for j in self.eq_set))
            xomx[k_temp:k_temp+x_r1_2.shape[0],:] = x_r1_2
            xomy[k_temp:k_temp+y_r1_2.shape[0],:] = y_r1_2
            k_temp += x_r1_2.shape[0]
            
        det1 = 0.
        n_iter = 0
        
        #Start of main loop
        while np.abs(det1-det)>epsilon and n_iter<=maxiter:
            n_iter += 1            
            det = det1
            sig_i = sig1
            sig1inv_i = sig1inv
            xomxi = np.zeros((self.n_eq*self.k,self.n_eq*self.k),float)
            xomyi = np.zeros((self.n_eq*self.k,1),float)
            r_temp = 0
            for r in range(self.n_eq):
                j_temp = 0
                for j in range(self.n_eq):
                    xomxi[r_temp:r_temp+self.k,j_temp:j_temp+self.k] = xomx[r_temp:r_temp+self.k,j_temp:j_temp+self.k]*sig1inv_i[r,j]
                    xomyi[r_temp:r_temp+self.k,0] = xomyi[r_temp:r_temp+self.k,0]+xomy[r_temp:r_temp+self.k,j]*sig1inv_i[r,j]
                    j_temp += self.k
                r_temp += self.k
            try:
                xomix = la.inv(xomxi)
            except:
                raise Exception, "ERROR: singular variance matrix"
            bml1 = np.dot(xomix,xomyi)

            #New residuals
            k_temp = 0
            Bigyh = np.zeros((self.n,self.n_eq),float)
            for r in range(self.n_eq):
                Bigyh[:,r] = np.dot(results_stp2[self.eq_set[r]].x,bml1[k_temp:k_temp+self.k,0])
            BigE = BigY-Bigyh

            #New sigma
            sig1 = np.dot(BigE.T,BigE)
            try:
                sig1inv = la.inv(sig1)
            except:
                raise Exception, "ERROR: singular variance matrix"

            sig1 = sig1/self.n
            sig1inv = sig1inv*self.n
            det1 = la.det(sig1inv)
            #End of main loop

        if n_iter > maxiter:
            raise Exception, "Convergence not achieved in maximum iterations" #transform into warn

        self.betas = bml1
        self.vm = xomix
        sigSUR = sig1
        sigSURi = sig1inv
        sig_diag = sigSUR.diagonal().reshape(1,self.n_eq)
        self.u_cov = sigSUR/np.sqrt(np.dot(sig_diag.T,sig_diag))

        # log-likelihood
        #ldet = np.log(la.det(sig_i))
        ldet = det1
        n2 = self.n/2.
        lik = -(n2*self.n_eq)*(np.log(2.*np.pi)+1.)-(n2*ldet)
        
        #Prepare output
        self.multi = results_stp2
        k_b = 0
        for r in range(self.n_eq):
            k_b1 = self.multi[self.eq_set[r]].betas.shape[0]
            self.multi[self.eq_set[r]].betas[:,] = self.betas[k_b:k_b+k_b1,]
            self.multi[self.eq_set[r]].vm = self.vm[k_b:k_b+k_b1,k_b:k_b+k_b1]
            k_b += k_b1
            self.multi[self.eq_set[r]].predy = Bigyh[:,r]
            self.multi[self.eq_set[r]].u = BigE[:,r]
            self.multi[self.eq_set[r]].logl = lik

        title = "ML SUR - EQUATION "            
        self.multi = USER.set_name_multi(self.multi,self.eq_set,name_equationID,y,x,name_y,name_x,name_ds,title,name_w,robust=None,endog=False,sp_lag=False)
        SUMMARY.OLS_multi(reg=self, multireg=self.multi, vm=vm, nonspat_diag=False, spat_diag=False, moran=False, sur=True)

class ML_Error_SUR(_sur_frame):
    def __init__(self, y, x, equationID, w,\
                 cores=None, name_y=None, name_x=None,\
                 name_equationID=None, name_ds=None, name_w=None, vm=False,\
                 maxiter=1000, epsilon=0.00001):
        
        eq_ids, results_stp2 = _sur_frame.__init__(self, y, x, equationID, w, cores)
        
        #Building stacked matrices:
        BigU1 = np.hstack((results_stp2[r].u for r in self.eq_set))
        BigY = np.hstack((results_stp2[r].y for r in self.eq_set))

        #Running SUR:
        xw = dict((r,lag_spatial(w,results_stp2[r].x)) for r in self.eq_set)
        BigWY = lag_spatial(w,BigY)
        #wroot = SP.linalg.eigs(w.sparse)
        k_tot = self.n_eq*self.k
        XX = np.zeros((k_tot,k_tot),float)
        WXX = np.zeros((k_tot,k_tot),float)
        XWX = np.zeros((k_tot,k_tot),float)
        XWWX = np.zeros((k_tot,k_tot),float)

        XY = np.zeros((k_tot,self.n_eq),float)
        WXY = np.zeros((k_tot,self.n_eq),float)
        XWY = np.zeros((k_tot,self.n_eq),float)
        XWWY = np.zeros((k_tot,self.n_eq),float)

        r_temp = 0
        for r in range(self.n_eq):
            xi = results_stp2[self.eq_set[r]].x
            wxi = xw[self.eq_set[r]]
            j_temp = 0
            for j in range(self.n_eq):
                xj = results_stp2[self.eq_set[j]].x
                wxj = xw[self.eq_set[j]]
                XX[r_temp:r_temp+self.k,j_temp:j_temp+self.k] = spdot(xi.T,xj)
                WXX[r_temp:r_temp+self.k,j_temp:j_temp+self.k] = spdot(wxi.T,xj)
                XWX[r_temp:r_temp+self.k,j_temp:j_temp+self.k] = spdot(xi.T,wxj)
                XWWX[r_temp:r_temp+self.k,j_temp:j_temp+self.k] = spdot(wxi.T,wxj)
                XY[r_temp:r_temp+self.k,j] = spdot(xi.T,BigY[:,j])
                WXY[r_temp:r_temp+self.k,j] = spdot(wxi.T,BigY[:,j])
                XWY[r_temp:r_temp+self.k,j] = spdot(xi.T,BigWY[:,j])
                XWWY[r_temp:r_temp+self.k,j] = spdot(wxi.T,BigWY[:,j])
                j_temp += self.k
            r_temp += self.k
 
        #Start of main loop
        n_iter = 0
        L1 = 0
        lambd1 = 0.5*np.ones((self.n_eq,1),float)

        while np.abs(lambd1-lambd)>epsilon and n_iter<=maxiter:
            n_iter += 1
            lambd = lambd1
            L0 = L1
            
            ulam = BigU1 - lag_spatial(w,BigU1)*lambd1.T
            sig = np.dot(ulam.T,ulam) / self.n
            try:
                sigi = la.inv(sig)
                det1 = determinant(sigi)
            except:        
                raise Exception, "ERROR: singular variance matrix"

            # FGLS for betas
            xomxi = np.zeros((k_tot,k_tot),float)
            xomyi = np.zeros((k_tot,1),float)

            r_temp = 0
            for r in range(self.n_eq):
                r_temp2 = r_temp+self.k
                lami=lambd[r][0]
                j_temp = 0
                for j in range(self.n_eq):
                    j_temp2 = j_temp+self.k
                    lamj=lambd[j][0]
                    xomxi[r_temp:r_temp2,j_temp:j_temp2] = sigi[r,j]*(XX[r_temp:r_temp2,j_temp:j_temp2] - lami*WXX[r_temp:r_temp2,j_temp:j_temp2] - lamj*XWX[r_temp:r_temp2,j_temp:j_temp2] + (lami*lamj)*XWWX[r_temp:r_temp2,j_temp:j_temp2])
                    xomyi[r_temp:r_temp2,0] = xomyi[r_temp:r_temp2,0]+sigi[r,j]*(XY[r_temp:r_temp2,j] - lami*WXY[r_temp:r_temp2,j] - lamj*XWY[r_temp:r_temp2,j] + (lami*lamj)*XWWY[r_temp:r_temp2,j])
                    j_temp += self.k
                r_temp += self.k

            try:
                xomix = la.inv(xomxi)
            except:        
                raise Exception, "ERROR: singular variance matrix"

            bml1 = np.dot(xomix,xomyi)





def _run_stp1(y,x,eq_ids,r):
    y_r = y[eq_ids[r]]
    x_r = x[eq_ids[r]]     
    x_constant = USER.check_constant(x_r)
    model = BaseOLS(y_r, x_constant)
    #model.logll = diagnostics.log_likelihood(model) 
    return model            

if __name__ == '__main__':
    #_test()          

    import pysal
    import numpy as np
    db = pysal.open(pysal.examples.get_path("NAT.dbf"),'r')
    y_var = ['HR80','HR90']
    y = np.vstack((np.array([db.by_col(r)]).T for r in y_var))
    y_name = 'HR'
    x_var = ['PS80','UE80','PS90','UE90']
    x = np.array([db.by_col(name) for name in x_var]).T
    x = np.vstack((x[:,0:2],x[:,2:4]))
    x_name = ['PS','UE']
    eq_name = 'Year'
    eq_ID = [1980]*3085 + [1990]*3085
    w = pysal.rook_from_shapefile(pysal.examples.get_path("NAT.shp"))
    w.transform = 'r'

    sur1=ML_SUR(y, x, eq_ID)
    print sur1.summary        
