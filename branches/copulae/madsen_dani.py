'''
Dani's part of porting Madsen's code
'''

import numpy as np
from scipy.linalg import inv, det
from scipy.stats import norm

def negLogEL(theta, y, U, XX, H, dimbeta, B, Bphi):
    '''
    Porting of Negative Log Exponential Likelihood from Madsen's MATLAB
    '''
    if np.isinf(theta[0]):
        alphaN = 1
    else:
        alphaN = np.exp(theta[0] / (1 + np.exp(theta[0])))
    if np.isinf(np.exp(theta[1])):
        alphaR = B
    else:
        alphaR = B * np.exp(theta[1]) / (1 + np.exp(theta[1]))
    if np.isinf(np.exp(theta[2])):
        phi = Bphi
    else:
        phi = Bphi * np.exp(theta[2]) / (1 + np.exp(theta[2]))
    betanew = theta[3:]

    Sigma = alphaN * np.exp(-H * alphaR)
    for ij in np.nonzero(H==0):
        Sigma[ij] = 1.
    SigmaInv = inv(Sigma)
    logdetSigma = np.sum(log(det(Sigma)))
    mu = np.exp(np.dot(XX, betanew))

    n, m = U.shape

    F = np.zeros((n, m))
    z = np.zeros((n, m))
    dFdphi = np.zeros((n,m))
    dFdmu = np.zeros((n,m))
    dzdphi = np.zeros((n,m))
    dzdmu = np.zeros((n,m))

    for i in range(n):
        F[i, :] = np.array([np.sum(p(range(y[i] - 1), phi, mu[i]))] * m) \
                 + U[i, :] * p(y[i], phi, mu[i])
        z[i, :] = np.array([norm.ppf(j, 0) for j in F[i, :]])
        dFdphi[i, :] = np.array([np.sum(dpdphi(range(y[i] - 1), phi, mu[i]))] * m) \
                + U[i, :] * dpdphi(y[i], phi, mu[i])
        dFdmu[i, :] = np.array([np.sum(dpdmu(range(y[i] - 1), phi, mu[i]))]) * m \
                + U[i, :] * dpdmu(y[i], phi, mu[i])
        dzdphi[i, :] = dFdphi[i, :] / norm.pdf(z[i, :])
        dzdmu[i, :] = dFdmu[i, :] / norm.pdf(z[i, :])

    dzdbeta = np.zeros((n, m, dimbeta))
    dmudbeta = np.zeros((n, dimbeta))
    for i in range(dimbeta):
        dmudbeta[:, i] = mu * XX[:,i]
        dzdbeta[:,:,i] = dzdmu * np.array([dmudbeta(:,i)] * m);

    T=np.zeros((1,m),float) 
    for j in range(m):
        T[j]=np.exp(-0.5*np.dot(z[:,j],np.dot((SigmaInv-np.eye(n)),z[:,j])))
    meanT=np.mean(T)

    # Calculate the negative log expected likelihood.
    NLEL=1./2.*logdetSigma-sum(np.log(p(y,phi,mu)))-np.log(meanT)

    if nargout > 1 % If derivatives are requested, calculate them.
    # Finish lines 93 - 121

    return nlel, eg

# p(y,phi,mu) is the negative binomial probability mass function with parameters phi and mu.
def p(y,phi,mu):
    p=np.zeros((y.shape),float)
    if mu.shape[0]==1:
        mu=mu*np.ones((y.shape))
    for i in range(y.shape[0]):
        if y[i]==0:
            p[i]=(phi**2./(1+phi**2))**(phi**2.*mu[i])
        else:
            p[i]=1./(y[i]*beta(y[i],phi**2.*mu[i]))*(phi**2/(1+phi**2))**(phi**2.*mu[i])*(1./(1+phi**2)**y[i]) #beta?
    return p

# dpdphi(y,phi,mu) is the derivative of p(y,phi,mu) with respect to phi.
def dpdphi(y,phi,mu):
    dpdphi=np.zeros((y.shape),float)
    if mu.shape[0]==1:
        mu=mu*np.ones((y.shape))
    for i in range(y.shape[0]):
        phi2 = phi**2.
        prodphi = phi2/(1+phi2)
        if y[i]==0:
            fac1 = prodphi**(phi2*mu[i])
            fac2 = (2.*phi*mu[i]*np.log(prodphi)+mu[i]*(2.*phi/(1+phi2)-2.*phi**3./(1+phi2)**2)*(1+phi2))
            fac3 = ((1+phi2)**y[i])-2.*prodphi**(phi2*mu[i])/((1+phi2)**y[i])*y[i]*phi/(1+phi2)
            dpdphi[i] = fac1*fac2/fac3
        else:
            fac1 = -1./y[i]/beta(phi2*mu[i],y[i])*prodphi**(phi2*mu[i])/((1+phi2)**y[i])
            fac2 = 2.*phi*mu[i]*psi(phi2*mu[i])-2.*phi*mu[i]*psi(y[i]+phi2*mu[i])
            fac3 = 1./y[i]/beta(phi2*mu[i],y[i])*prodphi**(phi2*mu[i])
            fac4 = 2.*phi*mu[i]*np.log(prodphi)+mu[i]*(2.*phi/(1+phi2)-2.*phi**3./(1+phi2)**2)*(1+phi**2)
            fac5 = ((1+phi2)**y[i])-2./beta(phi2*mu[i],y[i])
            fac6 = prodphi**(phi2*mu[i])/((1+phi2)**y[i])*phi/(1+phi2)
            dpdphi[i] = fac1*fac2+fac3*fac4/fac5*fac6
    return dpdphi

# dpdmu(y,phi,mu) is the derivative of p(y,phi,mu) with respect to mu.
def dpdmu=dpdmu(y,phi,mu):           
    dpdmu=np.zeros((y.shape),float)    
    if mu.shape[0]==1:
        mu=mu*np.ones((y.shape))
    for i in range(y.shape[0]):
        phi2 = phi**2.
        prodphi = phi2/(1+phi2)
        if y[i]==0:
            dpdmu[i]=prodphi**(phi2*mu[i])*phi2*np.log(prodphi)
        else:
            fac1 = -1./y[i]/beta(y[i],phi2*mu[i])*prodphi**(phi2*mu[i])/((1+phi2)**y[i])
            fac2 = phi2*psi(phi2*mu[i])-phi2*psi(y[i]+phi2*mu[i])
            fac3 = 1./y[i]/beta(y[i],phi2*mu[i])*prodphi**(phi2*mu[i])
            fac4 = phi2*np.log(prodphi)/((1+phi2)**y[i])
            dpdmu[i] = fac1*fac2+fac3*fac4
    return dpdmu
