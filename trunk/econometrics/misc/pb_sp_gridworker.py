import sys, glob, pickle, struct, os, time, gzip
import numpy as np
import scipy.stats as stats
#Grid Worker:


def p_runs(R,data):
    N, V, B, scale1 = data
    sumPhi = 0
    for r in range(R):
        seed = abs(struct.unpack('i',os.urandom(4))[0])
        np.random.seed(seed)
        nn = np.zeros((N,1),float)
        vn = np.zeros((N,1),float)        
        sumbn = 0
        prodPhi = 1.0
        for i in range(N):
            n = -(i+1)
            vn[n] = 1.0*(V[n]-sumbn)/B[n,n]
            prodPhi = prodPhi * stats.norm.cdf(vn[n])
            if prodPhi < 1e-50 and scale1 > 0:
                prodPhi = prodPhi * 1e+200
                scale1 -= 1
            if i<N-1:
                nn[n] = np.random.normal(0,1)                
                tdraw = time.time()
                while nn[n] >= vn[n]:
                    nn[n] = np.random.normal(0,1)
                    if time.time() - tdraw > 15:
                        return 'Fail'
                sumbn = np.dot(B[n-1:n,n:],nn[n:])
        if scale1 > 0:
            for i in range(scale1):
                prodPhi = prodPhi * 1e+200
        sumPhi += prodPhi
    if sumPhi == 0:
        return 'Fail'
    return float(sumPhi)


"""
Grid helper functions
"""

def get_grid(data,cycles,IDs):    
    inpath = 'misc/probit_sp.pklz'
    infile = gzip.open(inpath, 'wb')
    pickle.dump(data, infile, -1)
    infile.close()
    xgrid_ids = run_grid(cycles,IDs) # call grid loader    
    repeat = grid_results(xgrid_ids,IDs)
    #Check which ones are done
    while repeat:
        done = []
        fname = 'misc/run_*.pkl'
        for i in glob.glob(fname): 
            i = i.split('.')
            i = i[0].split('/')
            i = i[1].split('_')
            i = int(i[1])
            done.append(i)        
        for i in IDs: #Enumerate which were not done,
            if done.count(i)>0:
                repeat.remove(i)
        if repeat: #and re-send them.
            print 'Repeating:', repeat
            xgrid_ids = run_grid(cycles,repeat)
            repeat = grid_results(xgrid_ids,IDs)           
    output = []
    for i in range(len(IDs)):
        outfile = 'misc/run_%s.pkl' %i
        pkl_file = open(outfile, 'rb')
        data = pickle.load(pkl_file)
        pkl_file.close()
        output.append(data)
        #print data
        os.remove(outfile)
    os.remove(inpath)
    return output

def run_grid(cycles,IDs): #Send jobs
    import grid_loader as gl
    xgrid_ids = []
    for i in IDs:    
        args = ' %s %s misc/' %(cycles,i)
        jid = gl.load_python('misc/pb_sp_gridworker.py',args=args)
        print 'loaded', jid, i
        xgrid_ids.append([jid, i])
    print 'all files loaded\n'
    return xgrid_ids

def grid_results(xgrid_ids,IDs):
    import grid_loader as gl
    xgrid_ids = gl.results_runner(xgrid_ids,'misc',delay=5)
    repeat = []
    for i in IDs:
        repeat.append(i)
    time.sleep(3)
    return repeat

if __name__ == '__main__':
    """
    Evaluates the boundaries and gets the sum over R of the product of the cumulative distributions.
    """
    args = sys.argv
    R = int(args[1])
    R_id = int(args[2])
    infile = args[3]+'/probit_sp.pklz'
    pkl_file = gzip.open(infile, 'rb')
    data = pickle.load(pkl_file)
    pkl_file.close()

    sump = p_runs(R,data)

    outfile = 'run_%s.pkl' %R_id
    output = open(outfile, 'wb')
    pickle.dump(sump, output, -1)
    output.close()

