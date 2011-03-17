import pysal
import numpy as np
from madsen import negLogEL

a = []
txt_link = 'matlab/LocOMGrubs.txt'
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

beta0 = np.array([[-25.3649],[12.5260],[-1.9425],[0.0936]])

y = a[:, 3]
muY = np.exp(np.dot(XX, beta0))
vy = (y - muY)**2

phi0 = 4.3478

theta0 = [np.log(alphaN0/(1-alphaN0)), np.log((alphaR0/B)/(1-alphaR0/B)), np.log((phi0/Bphi)/(1-phi0/Bphi)), beta0]
dimbeta = beta0.shape[0]

numu = 2
#U = np.random.uniform(0, 1, (n, numu))
#np.savetxt('U.txt',U,delimiter=",")
U = np.loadtxt('U.txt',delimiter=",")

l = negLogEL(theta0, y, U, XX, H, dimbeta, B, Bphi, want_derivatives=1)
print '$$$$$$$$$$$$$$$$$$$$$$'
print l
