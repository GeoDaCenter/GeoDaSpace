import pysal
import numpy as np
from madsenD import negLogEL, p, dpdphi, dpdmu
from scipy.optimize import fmin_l_bfgs_b
import scipy.optimize as op
import scikits.statsmodels.api as sm

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
#B = 0.1958
mH = min([i for i in H.flatten() if i!=0.])
B = -np.log(.05) / mH
Bphi = 100

XX = a[:, 2:3]
XX = np.hstack((np.ones((n, 1)), XX, XX**2, XX**3))
alphaN0 = 0.2
#alphaR0 = 0.1052
#alphaR0 = -np.log(.1 / alphaN0) / mH

#beta0 = np.array([[-25.3640],[12.5260],[-1.9425],[0.0936]])
beta0 = sm.GLM(a[:, 3], XX[:, :], family=sm.families.Poisson())
beta0 = beta0.fit().params

y = a[:, 3]
muY = np.exp(np.dot(XX, beta0))
vy = (y - muY)**2

phi0 = 4.3478
# Alternatives
ratio = muY / (vy - muY)
#phi0 = min(abs(ratio))
#phi0 = np.mean(abs(ratio))
#phi0 = max(abs(ratio))

# Create (real) W
#w = pysal.lat2W(2,71)
#w = pysal.knnW(xy, k=40)
w = pysal.threshold_continuousW_from_array(xy, 1000)
w.transform='r'
H = w.full()[0]

theta0 = []
#theta0.append(np.log(alphaN0/(1-alphaN0)))
#theta0.append(np.log((phi0/Bphi)/(1-phi0/Bphi)))
theta0.append(alphaN0)
theta0.append(phi0)
for i in beta0:
    theta0.append(float(i))
dimbeta = beta0.shape[0]
numu = 10
#U = np.random.uniform(0, 1, (n, numu))
#np.savetxt('U1k.txt',U,delimiter=",")
U = np.loadtxt('U.txt',delimiter=",")
bounds=[(-0.99999,0.99999),(None,Bphi),(None,None),(None,None),(None,None),(None,None)]
bounds=[(-1.,1.),(None,Bphi),(None,None),(None,None),(None,None),(None,None)]
#print 'par_hat', negLogEL(np.array(theta0), y, U, XX, H, dimbeta, Bphi, want_derivatives=1)
use_derivatives = 1
if use_derivatives:
    ll_func = lambda par: negLogEL(par, y, U, XX, H, dimbeta, Bphi, want_derivatives=1)
    par_hat = fmin_l_bfgs_b(ll_func, np.array(theta0), iprint=1, bounds=bounds, approx_grad=0)
else:
    ll_func = lambda par: negLogEL(par, y, U, XX, H, dimbeta, Bphi, want_derivatives=0)
    par_hat = fmin_l_bfgs_b(ll_func, np.array(theta0), iprint=1, bounds=bounds, approx_grad=1)


#par_hat = fmin_l_bfgs_b(ll_func, np.array(theta0), iprint=1)
#par_hat = op.fmin(ll_func, np.array(theta0), retall=True)

print '##################################################################'
print 'Initial estimates:\n',np.array(theta0)
print 'Estimates:\n', par_hat[0]
print '\n'
print 'NLEL:\t', par_hat[1]
print '\n'
for i in par_hat[2]:
    print i, ':\t', par_hat[2][i]
"""
print '\n'
print '### Coefs ###'
coefs = ['theta_1', 'theta_2 ', 'Phi     ', 'Beta_0 ', 'Beta_1 ', 'Beta_2 ', 'Beta_3 ']
for i in range(len(coefs)):
    line = '%s:\t%f'%(coefs[i], par_hat[0][i])
    print line
'''
print(dpdmu(np.array([0]), np.array([11.1004]), np.array([.6576])))
print(dpdmu(np.array([4]), np.array([11.1004]), np.array([1.3605])))
'''
"""
