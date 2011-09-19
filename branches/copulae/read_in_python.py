import pysal
import numpy as np
from madsen import negLogEL
from scipy.optimize import fmin_l_bfgs_b
import scipy.optimize as op

a = []
txt_link = 'LocOMGrubs.txt'
txt = open(txt_link)
for line in txt:
    line = line.strip('\n').strip('\r').split()
    line = map(float, line)
    a.append(line)
txt.close()
a = np.array(a)

xy = a[:, :2]
n = a.shape[0]
Zx = np.dot(xy[:, :1], np.ones(xy[:, :1].T.shape))
Zy = np.dot(xy[:, 1:2], np.ones(xy[:, 1:2].T.shape))
H = np.sqrt((Zx - Zx.T)**2 + (Zy - Zy.T)**2)
B = 0.1958
Bphi = 100

XX = a[:, 2:3]
XX = np.hstack((np.ones((n, 1)), XX, XX**2, XX**3))
alphaN0 = 0.5
alphaR0 = 0.1052

beta0 = np.array([[-25.3640],[12.5260],[-1.9425],[0.0936]])

y = a[:, 3]
muY = np.exp(np.dot(XX, beta0))
vy = (y - muY)**2

phi0 = 4.3478
theta0 = []
theta0.append(np.log(alphaN0/(1-alphaN0)))
theta0.append(np.log((alphaR0/B)/(1-alphaR0/B)))
theta0.append(np.log((phi0/Bphi)/(1-phi0/Bphi)))
for i in beta0:
    theta0.append(float(i))
#theta0 = [np.log(alphaN0/(1-alphaN0)), np.log((alphaR0/B)/(1-alphaR0/B)), np.log((phi0/Bphi)/(1-phi0/Bphi)), beta0]
dimbeta = beta0.shape[0]
numu = 10
U = np.random.uniform(0, 1, (n, numu))
#np.savetxt('U.txt',U,delimiter=",")
#U = np.loadtxt('U.txt',delimiter=",")
print np.array(theta0)
bounds=[(0.,1.),(None,B),(None,Bphi),(None,None),(None,None),(None,None),(None,None)]
par_hat = negLogEL(np.array(theta0), y, U, XX, H, dimbeta, B, Bphi, want_derivatives=1)
ll_func = lambda par: negLogEL(par, y, U, XX, H, dimbeta, B, Bphi, want_derivatives=1)
par_hat = fmin_l_bfgs_b(ll_func, np.array(theta0), iprint=1)
#par_hat = op.fmin(ll_func, np.array(theta0), retall=True)

print '$$$$$$$$$$$$$$$$$$$$$$'
print par_hat
