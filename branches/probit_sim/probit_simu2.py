import pysal
import struct
import os
import probit_dev as pb
import numpy as np
import scipy.sparse as SP
import time
import multiprocessing as mp

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
        if core == 'multi':
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
                    parts = pool.map(run_probit, [(x,I,lamb,w.sparse,n,b,R/cores)] * cores)
                    output = output_paster(parts)
                    if lamb == 0:
                        self.partes = parts
                        self.out = output
                if core == 'single':
                    output = run_probit([x,I,lamb,w.sparse,n,b,R]) #Run the probit estimation
                t2 = time.time()
                print "N:", n, "Lambda:", lamb, "Time:", t2-t1 #Prints follow-up messages
                self.get_results(output,counter,ll,l,lamb,b,t2-t1) #Stores the results in the result array
                t1 = t2 #Update the timer
                if lamb == 0:
                    tests_stats = (output[2][:,0],output[3][:,0],output[4][:,0])
                    save_test_null(tests_stats,counter,("testnull1.txt","testnull2.txt","testnull3.txt","testnull4.txt","testnull5.txt"))
            self.result[counter*ll:(counter+1)*ll,0] = n #Store the number of obs used for these estimations
            counter += 1
            #Save partial results for each of the first 4 sample sizes:
            files = ("resultp1.txt","resultp2.txt","resultp3.txt","resultp4.txt")
            if counter<5:
                np.savetxt(files[counter-1],self.result[0:counter*ll],delimiter=",")
            x0 = x #Update the base sample set for next round
        #Save all results:
        np.savetxt("result.txt",self.result,delimiter=",")
        if core == 'multi':
            R = R/cores*cores
        print 'Number of runs actually performed:', R

    def get_results(self,output,counter,ll,l,lamb,b,t): #Stores all the results in the 'result' array
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
        b        : array
                   Value of betas (kx1)
        t        : float
                   Time elapsed to get the results for this specific lambda and sample set
        """

        self.result[counter*ll+l,1] = lamb
        self.result[counter*ll+l,2] = np.mean(output[2][:,0]) #Moran's statistics
        self.result[counter*ll+l,3] = np.mean(output[2][:,1]) #Moran's rejection rate 5%
        self.result[counter*ll+l,3] = np.mean(output[2][:,2]) #Moran's rejection rate 10%
        self.result[counter*ll+l,4] = np.mean(output[3][:,0]) #Pinkse's statistics
        self.result[counter*ll+l,5] = np.mean(output[3][:,1]) #Pinkse's rejection rate 5%
        self.result[counter*ll+l,5] = np.mean(output[3][:,2]) #Pinkse's rejection rate 10%
        self.result[counter*ll+l,6] = np.mean(output[4][:,0]) #Pinkse_Slade's statistics
        self.result[counter*ll+l,7] = np.mean(output[4][:,1]) #Pinkse_Slade's rejection rate 5%
        self.result[counter*ll+l,5] = np.mean(output[3][:,2]) #Pinkse's rejection rate 10%
        self.result[counter*ll+l,-1] = t #Time used to computate all the runs for this sample set and lambda
        for k in range(b.shape[0]):
            dev = output[0][:,k]-b[k]
            self.result[counter*ll+l,8+k*3] = np.mean(output[0][:,k]) #Average of the estimated beta
            self.result[counter*ll+l,9+k*3] = 1.*sum(dev*dev)/output[0][:,k].shape[0]  #MSE of the estimated betas
            self.result[counter*ll+l,10+k*3] = np.mean(output[1][:,k]) #Average of the estimated standard deviation of beta

def run_probit(att):
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
    morans = np.zeros((cycles, 3),float) #KP Moran's I test statistic and rejection rate from each run
    pinkse = np.zeros((cycles, 3),float) #Pinkse's LM test statistic and rejection rate from each run
    pslade = np.zeros((cycles, 3),float) #Pinkse and Slade's LM test statistic and rejection rate from each run
    for r in range(cycles):
        seed = abs(struct.unpack('i',os.urandom(4))[0])
        np.random.seed(seed)
        e = (I-lambd*w) * np.random.normal(0,1,(n,1)) #Build residuals vector
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
            morans[r,2] = 1
        elif abs(morans[r,0])>1.645: #Critical value for normal distribution (10%)
            morans[r,2] = 1
        if abs(pinkse[r,0])>3.841: #Critical value for chi-square distribution (5%)
            pinkse[r,1] = 1
            pinkse[r,2] = 1
        elif abs(pinkse[r,0])>2.705: #Critical value for chi-square distribution (10%)
            pinkse[r,2] = 1
        if abs(pslade[r,0])>3.841: #Critical value for chi-square distribution (5%)
            pslade[r,1] = 1
            pslade[r,2] = 1
        elif abs(pslade[r,0])>2.705: #Critical value for chi-square distribution (10%)
            pslade[r,2] = 1
    return (betas, sd, morans, pinkse, pslade)

def output_paster(values_cores):
    """
    Merges output from different cores in one object

    Parameters
    ----------

    values_cores    : list
                      List of arrays out of different cores
    """
    betas = []
    sd = []
    morans = []
    pinkse = []
    pslade = []
    for i in range(len(values_cores)):
        betas.append(values_cores[i][0])
        sd.append(values_cores[i][1])
        morans.append(values_cores[i][2])
        pinkse.append(values_cores[i][3])
        pslade.append(values_cores[i][4])
    betas = np.vstack([i for i in betas])
    sd = np.vstack([i for i in sd])
    morans = np.vstack([i for i in morans])
    pinkse = np.vstack([i for i in pinkse])
    pslade = np.vstack([i for i in pslade])
    return (betas, sd, morans, pinkse, pslade)

def save_test_null(stats,counter,files):
    result = np.hstack([np.reshape(i,(R,1)) for i in stats])
    np.savetxt(files[counter],result,delimiter=",")


if __name__ == '__main__':
    b = np.reshape(np.array([1,0.5]),(2,1)) #set value for \betas
    R = 10000 #set the number of runs. If not divisible by the number of cores for multi_core, it will be changed to the biggest divisible lesser than the amount provided.
    x0 = np.random.normal(-2,1,(25,1)) #set the X0
    #x0 = np.reshape(np.array([[1,2,3,-1,-2,-3,2,-4,4]]),(9,1)) #set the X0
    x0 = np.hstack((np.ones(x0.shape),x0)) #Add constant to X0
    N = 5 #Number of sets of sample size. The samples sizes are n_i = 5**{N_i*2}
    lambd = [-0.8,-0.5,-0.3,-0.1,-0.01,0,0.01,0.1,0.3,0.5,0.8] #set values for \lambda
    simu = probit_simu(b,R,N,x0,lambd,core='multi')
