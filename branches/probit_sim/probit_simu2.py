import pysal
import probit_dev as pb
import numpy as np
import scipy.sparse as SP
import time
import multiprocessing as mp

def output_paster(values_cores):
    """
    Merges output from different cores in one object

    Parameters
    ----------

    values_cores    : list
                      List of arrays out of different cores
    """
    return np.vstack([i for i in values_cores])

def run_probit(att): #att = [x,I,lambd,w,n,b,R]. Since it seemed from mp.Pool.map that only one attribute could be passed, all were saved in the same list
    """
    Runs the probit estimation of each set of sample sizes and lambda

    Parameters
    ----------

    att   : list
            List of all attributes required.
            x = array: values of X (nxk)
            I = sparse: identity matrix (nxn)
            lambd = float: value of lambda
            w = w: weights matrix
            n = float: number of obs
            b = array: value of betas (kx1)
            cycles = float: number of runs
    """
    x = att[0]
    I = att[1]
    lambd = att[2]
    w = att[3]
    n = att[4]
    b = att[5]
    cycles = att[6]
    betas = np.zeros((cycles, x.shape[1]),float) #To store the estimated betas from each run
    sd = np.zeros((cycles, x.shape[1]),float) #To store the estimated standard deviation of the betas from each run 
    morans = np.zeros((cycles, 2),float) #KP Moran's I test statistic and rejection rate from each run
    pinkse = np.zeros((cycles, 2),float) #Pinkse's LM test statistic and rejection rate from each run
    pslade = np.zeros((cycles, 2),float) #Pinkse and Slade's LM test statistic and rejection rate from each run
    for r in range(cycles):
        e = (I-lambd*w.sparse) * np.random.normal(0,1,(n,1)) #Buld residuals vector
        ys = np.dot(x,b) + e #Build y_{star}
        y = np.zeros((n,1),float) #Binary y
        for yi in range(len(y)):
            if ys[yi]>0:
                y[yi] = 1
        probit1 = pb.probit(x,y,constant=False,w=w)
        sd0 = np.sqrt(probit1.vm.diagonal()) #Get estimated standard deviation from the VC matrix
        for k in range(betas.shape[1]):
            betas[r,k] = probit1.betas[k]
            sd[r,k] = sd0[k]
        morans[r,0] = probit1.moran
        pinkse[r,0] = probit1.LM_error
        pslade[r,0] = probit1.pinkse_slade
        if abs(morans[r,0])>1.96: #Critical value for normal distribution (5%)
            morans[r,1] = 1
        if abs(pinkse[r,0])>3.841: #Critical value for chi-square distribution (5%)
            pinkse[r,1] = 1
        if abs(pslade[r,0])>3.841: #Critical value for chi-square distribution (5%)
            pslade[r,1] = 1
    return [betas, sd, morans, pinkse, pslade]

class probit_simu:
    """
    Calls the probit estimation of all sets of sample sizes and lambdas and save the results

    Parameters
    ----------

    b     : array
            Value of betas (kx1)
    R     : float
            Number of runs
    N     : float
            Number of sample sets
    x0    : array
            Values of X (nxk)
    lambd : list
            Values of lambda
    core  : string
            'single'(default) if multiprocessing is not to be used
            'multi' if multiprocessing is to be used
    """

    def __init__(self,b,R,N,x0,lambd,core='single'):
        cores = mp.cpu_count()
        pool = mp.Pool(cores)
        t1 = time.time() #Start timer
        ll = len(lambd) #Number of different lamdas to be used
        self.result = np.zeros((N*ll,x0.shape[1]*3+9),float) #Start the results array. There are 3 columns for each x and 3 for each of the 3 tests
        counter = 0
        for i in (range(N+1)[1:]): #Prepare the estimations for each sample set
            w = pysal.lat2W(5**i,5**i)
            w.transform='r'
            n = 5**(i*2)
            x = x0 #Set the x equal to the base x0, which is updated at the end of the estimations for each sample set
            if counter > 0:
                for nn in range(24):
                    x = np.vstack((x,x0)) #New sample set 25 times bigger than the initial
            I = SP.lil_matrix(w.sparse.get_shape()) #Set sparse identity matrix
            I.setdiag(np.ones((n,1),float).flat)
            I = I.asformat('csr')
            for l in range(ll): #Run the estimation for each sample set and each lambda
                lamb = lambd[l]
                if core == 'multi':
                    parts = pool.map(run_probit, [x,I,lamb,w,n,b,R/cores] * cores)
                    output = map(output_paster,parts)
                if core == 'single':
                    output = run_probit([x,I,lamb,w,n,b,R/cores*cores]) #Run the probit estimation
                t2 = time.time()
                print "N:", n, "Lambda:", lamb, "Time:", t2-t1 #Prints follow-up messages
                self.get_results(output,counter,ll,l,lamb,x.shape[1],t2-t1) #Stores the results in the result array
                t1 = t2 #Update the timer
            self.result[counter*ll:(counter+1)*ll,0] = n #Store the number of obs used for these estimations
            counter += 1
            #Save partial results for each of the first 4 sample sizes:
            if counter == 1:
                np.savetxt("resultp1.txt",self.result[0:counter*ll],delimiter=",") 
            if counter == 2:
                np.savetxt("resultp2.txt",self.result[0:counter*ll],delimiter=",") 
            if counter == 3:
                np.savetxt("resultp3.txt",self.result[0:counter*ll],delimiter=",") 
            if counter == 4:
                np.savetxt("resultp4.txt",self.result[0:counter*ll],delimiter=",") 
            x0 = x #Update the base sample set for next round
        #Save all results:
        np.savetxt("result.txt",self.result,delimiter=",")
        print 'Number of runs actually performed:', R/cores*cores

    def get_results(self,output,counter,ll,l,lamb,K,t): #Stores all the results in the 'result' array
        """
        Runs the probit estimation of each set of sample sizes and lambda
    
        Parameters
        ----------

        output   : list
                   List with the results for each sample set and lambda
        counter  : float
                   Counter of sample sets
        ll       : float
                   Number of different lambdas
        l        : float
                   Counter of lamda
        lamb     : float
                   Value of lambda
        K        : float
                   Number of betas
        t        : float
                   Time elapsed to get the results for this specific lambda and sample set
        """

        self.result[counter*ll+l,1] = lamb
        self.result[counter*ll+l,2] = np.mean(output[2][:,0]) #Moran's statistics
        self.result[counter*ll+l,3] = np.mean(output[2][:,1]) #Moran's rejection rate
        self.result[counter*ll+l,4] = np.mean(output[3][:,0]) #Pinkse's statistics
        self.result[counter*ll+l,5] = np.mean(output[3][:,1]) #Pinkse's rejection rate
        self.result[counter*ll+l,6] = np.mean(output[4][:,0]) #Pinkse_Slade's statistics
        self.result[counter*ll+l,7] = np.mean(output[4][:,1]) #Pinkse_Slade's rejection rate
        self.result[counter*ll+l,-1] = t #Time used to computate all the runs for this sample set and lambda
        for k in range(K):
            self.result[counter*ll+l,8+k*3] = np.mean(output[0][:,k]) #Average of the estimated beta
            self.result[counter*ll+l,9+k*3] = np.std(output[0][:,k]) #Standard deviation of the estimated betas
            self.result[counter*ll+l,10+k*3] = np.mean(output[1][:,k]) #Average of the estimated standard deviation of beta


if __name__ == '__main__':
    b = np.reshape(np.array([-1,0.5]),(2,1)) #set value for \betas
    R = 20 #set the number of runs. If not divisible by the number of cores, it will be changed to the biggest divisible lesser than the amount provided.
    x0 = np.random.normal(2,1,(25,1)) #set the X0
    x0 = np.hstack((np.ones(x0.shape),x0)) #Add constant to X0
    N = 2 #Number of sets of sample size. The samples sizes are n_i = 5**{N_i*2}
    lambd = [-0.8,-0.5,-0.3,-0.1,-0.01,0,0.01,0.1,0.3,0.5,0.8] #set values for \lambda
    simu = probit_simu(b,R,N,x0,lambd)
