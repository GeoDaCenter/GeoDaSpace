'''
Madsen's code
'''

import numpy as np
from scipy.linalg import inv, det
from scipy.stats import norm, nbinom
from scipy.special import beta, psi

def negLogEL(theta, y, U, XX, H, dimbeta, Bphi, want_derivatives=None):
    '''
    Porting of Negative Log Exponential Likelihood from Madsen's MATLAB
    '''
    if np.isinf(np.exp(theta[0])):
        alphaN = 0.9
    else:
        alphaN = np.exp(theta[0]) / (1 + np.exp(theta[0]))    
    alphaN = theta[0]   
    lambd = alphaN
    if np.isinf(np.exp(theta[1])):
        phi = Bphi
    else:
        phi = Bphi * np.exp(theta[1]) / (1 + np.exp(theta[1]))    
    phi = theta[1]
    betanew = np.reshape(np.array(theta[2:]),(dimbeta,1))
    n, m = U.shape
    sig0 = np.eye(n) - alphaN*H
    SigmaInv = np.dot(sig0.T,sig0)
    Sigma = inv(SigmaInv)
    #print SigmaInv.diagonal()
    logdetSigma = np.log(1./det(SigmaInv))
    #print det(SigmaInv), logdetSigma
    print alphaN, phi, betanew.T
    mu = np.exp(np.dot(XX, betanew))
    F = np.zeros((n, m))
    z = np.zeros((n, m))
    dFdphi = np.zeros((n,m))
    dFdmu = np.zeros((n,m))
    dzdphi = np.zeros((n,m))
    dzdmu = np.zeros((n,m))

    for i in range(n):
        Ui = np.reshape(U[i, :],(1,m))
        #F[i, :] = np.array([np.sum(p(np.array(range(int(round(y[i])))), phi, np.array(mu[i])))] * m) \
        #         + Ui * p(np.array([y[i]]), phi, np.array(mu[i]))
        aaaa = np.array([np.sum(p(np.array(range(int(round(y[i])))), phi, np.array(mu[i])))] * m) \
                 + Ui * p(np.array([y[i]]), phi, np.array(mu[i]))
        aaaa[aaaa>0.99999999] = 0.99999999 
        F[i, :] = aaaa
        #print '$$$', np.sum(p(np.array(range(int(round(y[i])))), phi, np.array(mu[i]))),p(np.array([y[i]]), phi, np.array(mu[i])),Ui,y[i],phi,mu[i]
        #print F[i, :]
        z[i, :] = np.array([norm.ppf(j, 0) for j in F[i, :]])
        dFdphi[i, :] = np.array([np.sum(dpdphi(np.array(range(int(round(y[i])))), phi, np.array(mu[i])))] * m) \
                + Ui * dpdphi(np.array([y[i]]), phi, np.array(mu[i]))
        dFdmu[i, :] = np.array([np.sum(dpdmu(np.array(range(int(round(y[i])))), phi, np.array(mu[i])))] * m) \
                + Ui * dpdmu(np.array([y[i]]), phi, np.array(mu[i]))
        dzdphi[i, :] = dFdphi[i, :] / norm.pdf(z[i, :])
        dzdmu[i, :] = dFdmu[i, :] / norm.pdf(z[i, :])
    dzdbeta = np.zeros((n, m, dimbeta))
    dmudbeta = np.zeros((n, dimbeta))
    for i in range(dimbeta):
        dmudbeta[:, i] = np.reshape(np.multiply(mu, np.reshape(XX[:, i], (n, 1))),(n,))
        dzdbeta[:,:,i] = dzdmu * np.array([dmudbeta[:,i]] * m).T
    T=np.zeros((1,m),float) 
    for j in range(m):
        zj = np.reshape(z[:,j], (n, 1))
        T[0, j]=np.exp(-0.5*np.dot(zj.T,np.dot((SigmaInv-np.eye(n)),zj)))
    meanT=np.mean(T)
    #print zj.T
    # Calculate the negative log expected likelihood.
    NLEL=1./2.*logdetSigma-sum(np.log(p(y,phi,mu)))-np.log(meanT)
    print "logdetSigma:",logdetSigma,"p(y,phi,mu):",sum(np.log(p(y,phi,mu))),"np.log(meanT)",np.log(meanT)
    if np.isnan(np.log(meanT)):
        print z, F
    print "$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"
    print NLEL
    print "$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"

    #if nargout > 1 % If derivatives are requested, calculate them
    eg = None
    if want_derivatives:
        dSdlambda = -H -H.T + 2*alphaN*np.dot(H.T,H)
        dTdlambda = np.zeros((1, m))
        dTdbeta = np.zeros((dimbeta, m))
        dTdphi = np.zeros((1, m))
        dldbeta = np.zeros((1,dimbeta))
        SigmaInv1 = SigmaInv - np.eye(n)
        for j in range(m):
            zj = np.reshape(z[:,j], (n, 1))            
            exp_a = np.exp(np.dot(np.dot(zj.T,SigmaInv1), zj)/-2.)
            dTdlambda = exp_a - (np.dot(np.dot(zj.T,dSdlambda),zj)/-2.)
            dTdphi[0, j] = T[0, j]*np.dot(np.reshape(dzdphi[:, j], (1, n)), np.dot(SigmaInv1, zj))
            for i in range(dimbeta):
                dTdbeta[i, j] = T[0, j]*np.dot(np.reshape(dzdbeta[:, j, i],(1,n)), \
                         np.dot(SigmaInv1, zj))
        tr_dldlambda = np.sum(np.diagonal(np.dot(dSdlambda, Sigma)))
        dldlambda = tr_dldlambda/2. + np.mean(dTdlambda)/meanT
        dldphi = (np.mean(dTdphi) / meanT) - np.sum(dpdphi(y, phi, mu) / \
                p(y, phi, mu))
        for i in range(dimbeta):
            pdpdmumu = np.divide(np.multiply(np.reshape(dpdmu(y, phi, mu),(n,1)), mu),np.reshape(p(y,phi,mu),(n,1)))
            dldbeta[0, i] = (np.mean(dTdbeta[i, :].T) / meanT) - \
                    np.sum(np.multiply(pdpdmumu, np.reshape(XX[:, i],(n,1))))
        eg = []
        eg0 = -dldlambda * lambd*(1-lambd)
        eg.append(eg0)
        eg2 = dldphi * phi * (1 - phi / Bphi)
        eg.append(eg2)
        for i in dldbeta.T:
            eg.append(float(i))
    return NLEL, np.array(eg)

# p(y,phi,mu) is the negative binomial probability mass function with parameters phi and mu.
def p(y,phi,mu): #Madsen's code
    p=np.zeros((y.shape),float)
    if mu.shape[0]==1:
        mu=mu*np.ones((y.shape))
    for i in range(y.shape[0]):
        if y[i]==0:
            p[i]=(phi**2./(1+phi**2))**(phi**2.*mu[i])
        else:
            p[i]=1./(y[i]*beta(y[i],phi**2.*mu[i]))*(phi**2/(1+phi**2))**(phi**2.*mu[i])*(1./(1+phi**2)**y[i])
    if sum(p)>0.9999:
        p[0]=0.9999
    return p

def p0(y,phi,mu):
    ps = np.zeros(y.shape)
    phi=phi**2.
    if mu.shape[0]==1:
        mu=mu*np.ones((y.shape))  
    for i in range(y.shape[0]):
        if y[i]==0:
            ps[i]=(phi/(1+phi))**(phi*mu[i])
        else:        
            r = phi*mu[i]
            bn = nbinom(r, (phi/(phi+1)))
            ps[i] = bn.pmf(y[i])
        if np.isnan(ps[i]):
            print "phi",phi,"mu",mu[i],"y",y[i]
    if sum(ps)==1:
        ps[0]=0.9999
    return ps


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
            dpdphi[i] = -1./y[i]/beta(phi2*mu[i],y[i])*prodphi**(phi2*mu[i])/((1+phi2)**y[i])*(2.*phi*mu[i]*psi(phi2*mu[i])-2.*phi*mu[i]*psi(y[i]+phi2*mu[i]))+1./y[i]/beta(phi2*mu[i],y[i])*prodphi**(phi2*mu[i])*(2.*phi*mu[i]*np.log(prodphi)+mu[i]*(2.*phi/(1+phi2)-2.*phi**3./(1+phi2)**2)*(1+phi2))/((1+phi2)**y[i])-2./beta(phi2*mu[i],y[i])*prodphi**(phi2*mu[i])/((1+phi2)**y[i])*phi/(1+phi2)
    return dpdphi

# dpdmu(y,phi,mu) is the derivative of p(y,phi,mu) with respect to mu.
def dpdmu(y,phi,mu):           
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


