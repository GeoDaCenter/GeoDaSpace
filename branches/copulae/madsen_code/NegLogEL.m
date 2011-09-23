%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% NegLogEL(theta,y,U,XX,H,dimbeta,B,Bphi)
%
% This function returns the negative log expected likelihood for a negative
% binomial marginal model. The derivative vector is returned as the second
% output argument.
%
% Input arguments:
% theta=current value of the parameter vector (alphaN,alphaR,phi,beta)
% y=data vector, assumed a column vector of length n
% U=matrix of independent uniform (0,1) RVs. Dimension of U is nxm where n
%   is the length of data vector n and m is the number of "jitters" to
%   average over.
% XX=nxp design matrix where p is the dimension of beta.
% H=nxn spatial distance matrix for the distance between observations in y.
% dimbeta=dimension of beta.
% B,Bphi=bounds on alphaR and phi to prevent overflow during the
%   optimization.
%
% Required library:
% specfun (by Paul Godfrey, available at my.fit.edu/~gabdo/paulbio.html)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [NLEL,EG]=NegLogEL(theta,y,U,XX,H,dimbeta,B,Bphi)

% The vector theta contains current values of parameters
% theta=[alphaN,alphaR,phi,beta0,beta1,beta2,beta3].
if isinf(exp(theta(1)))
    alphaN=1; %If theta(1) is large, alphaN is approx. 1.
else
    alphaN=exp(theta(1))/(1+exp(theta(1))); %alphaN is correlation "nugget" and must be in (0,1).
end

% Both alphaR and phi are constrained for numerical stability. These
% constraints are accomplished by the following transformations.
if isinf(exp(theta(2)))
    alphaR=B;
else
    alphaR=B*exp(theta(2))/(1+exp(theta(2))); 
end;
if isinf(exp(theta(3)))
    phi=Bphi;
else
    phi=Bphi*exp(theta(3))/(1+exp(theta(3))); 
end;
betanew=theta(4:length(theta));

% Calculate spatial covariance matrix from distance matrix H and spatial
% parameters alphaN and alphaR.
Sigma=alphaN*exp(-H*alphaR);
Sigma(find(H==0))=1;
SigmaInv=Sigma^(-1);
logdetSigma=sum(log(eig(Sigma))); % Calculate (log(det(Sigma))).
mu=exp(XX*betanew'); % Calculate current mean vector.

% Determine the length of the data vector and the number of random jitters
% to use.
[n m]=size(U);

% Initialize variables.
F=zeros(n,m); % F(i,j) is F*_i(y_i-u_{i,j})
z=zeros(n,m); % z(i,j) is Phi^(-1)(F(i,j)).
dFdphi=zeros(n,m); % dFdphi(i,j) is the derivative of F(i,j) with respect to phi.
dFdmu=zeros(n,m); % dFdmu(i,j) is the derivative of F(i,j) with respect to mu.
dzdphi=zeros(n,m); % dzdphi(i,j) is the derivative of z(i,j) with respect to phi.
dzdmu=zeros(n,m); % dzdmu(i,j) is the derivative of z(i,j) with respect to mu.

% Calculate F, z, and derivatives dzdphi and dzdmu.
for i=1:n 
    F(i,:)=repmat(sum(p(0:y(i)-1,phi,mu(i))),1,m)+U(i,:)*p(y(i),phi,mu(i)); 
    z(i,:)=norminv(F(i,:),0,1);
    dFdphi(i,:)=repmat(sum(dpdphi(0:y(i)-1,phi,mu(i))),1,m)+U(i,:)*dpdphi(y(i),phi,mu(i)); %checked
    dFdmu(i,:)=repmat(sum(dpdmu(0:y(i)-1,phi,mu(i))),1,m)+U(i,:)*dpdmu(y(i),phi,mu(i)); %checked
    dzdphi(i,:)=dFdphi(i,:)./normpdf(z(i,:)); 
    dzdmu(i,:)=dFdmu(i,:)./normpdf(z(i,:));
end

% Calculate derivatives dzdbeta and dmudbeta.
dzdbeta=zeros(n,m,dimbeta);
dmudbeta=zeros(n,dimbeta);
for i=1:dimbeta
    dmudbeta(:,i)=mu.*XX(:,i); %checked.
    dzdbeta(:,:,i)=dzdmu.*repmat(dmudbeta(:,i),1,m);
end

% T is the mx1 vector of exponentials in the log expected likelihood.
T=zeros(1,m); 
for j=1:m T(j)=exp(-1/2*z(:,j)'*(SigmaInv-eye(n))*z(:,j)); end;
meanT=mean(T);
% Calculate the negative log expected likelihood.
NLEL=1/2*logdetSigma-sum(log(p(y,phi,mu)))-log(meanT);

if nargout > 1 % If derivatives are requested, calculate them.
    dSigmadalphaR=-alphaN*H.*exp(-H*alphaR);
    dSigmadalphaR(find(H==0))=0;
    dSigmadalphaN=exp(-H*alphaR);
    dSigmadalphaN(find(H==0))=0;
    
    dTdalphaR=zeros(1,m);
    dTdalphaN=zeros(1,m);
    dTdbeta=zeros(dimbeta,m);
    dTdphi=zeros(1,m);
    dldbeta=zeros(1,dimbeta);
   
    for j=1:m
        dTdalphaR(j)=exp(-1/2*z(:,j)'*(SigmaInv-eye(n))*z(:,j))*1/2*z(:,j)'*SigmaInv*dSigmadalphaR*SigmaInv*z(:,j);
        dTdalphaN(j)=exp(-1/2*z(:,j)'*(SigmaInv-eye(n))*z(:,j))*1/2*z(:,j)'*SigmaInv*dSigmadalphaN*SigmaInv*z(:,j);
        dTdphi(j)=T(j)*dzdphi(:,j)'*(SigmaInv-eye(n))*z(:,j);
        for i=1:dimbeta
            dTdbeta(i,j)=T(j)*dzdbeta(:,j,i)'*(SigmaInv-eye(n))*z(:,j);
        end % for i
    end % for j
    
    dldalphaR=1/2*trace(SigmaInv*dSigmadalphaR)- 1/meanT*mean(dTdalphaR);
    dldalphaN=1/2*trace(SigmaInv*dSigmadalphaN)- 1/meanT*mean(dTdalphaN);
    dldphi=mean(dTdphi)/meanT-sum(dpdphi(y,phi,mu)./p(y,phi,mu));
    for i=1:dimbeta dldbeta(i)=mean(dTdbeta(i,:)')/meanT-sum(dpdmu(y,phi,mu).*mu.*XX(:,i)./p(y,phi,mu)); end;
    %If alpha is transformed, dldalpha is (derivative wrt transformed alpha)*(derivative of transformation evaluated at untransformed value).
    EG=[dldalphaN*alphaN*(1-alphaN) dldalphaR*alphaR*(1-alphaR/B) dldphi*phi*(1-phi/Bphi) dldbeta]'; 

end

% p(y,phi,mu) is the negative binomial probability mass function with
% parameters phi and mu.
function p=p(y,phi,mu)
p=zeros(size(y));
if length(mu)==1 mu=mu*ones(size(y)); end;
ix=find(y==0);
p(ix)=(phi.^2./(1+phi.^2)).^(phi.^2.*mu(ix));
ix=find(y>0);
p(ix)=1./(y(ix).*beta(y(ix),phi.^2.*mu(ix))).*(phi.^2/(1+phi.^2)).^(phi.^2.*mu(ix)).*(1./(1+phi.^2).^y(ix));

% dpdphi(y,phi,mu) is the derivative of p(y,phi,mu) with respect to phi.
function dpdphi=dpdphi(y,phi,mu)
dpdphi=zeros(size(y));
if length(mu)==1 mu=mu*ones(size(y)); end;
ix=find(y==0);
dpdphi(ix)=(phi.^2./(1+phi.^2)).^(phi.^2.*mu(ix)).*(2.*phi.*mu(ix).*log(phi.^2./(1+phi.^2))+mu(ix).*(2.*phi./(1+phi.^2)-2.*phi.^3./(1+phi.^2).^2).*(1+phi.^2))./((1+phi.^2).^y(ix))-2.*(phi.^2./(1+phi.^2)).^(phi.^2.*mu(ix))./((1+phi.^2).^y(ix)).*y(ix).*phi./(1+phi.^2);
ix=find(y>0);
dpdphi(ix)=-1./y(ix)./beta(phi.^2.*mu(ix),y(ix)).*(phi.^2./(1+phi.^2)).^(phi.^2.*mu(ix))./((1+phi.^2).^y(ix)).*(2.*phi.*mu(ix).*Psi(phi.^2.*mu(ix))-2.*phi.*mu(ix).*Psi(y(ix)+phi.^2.*mu(ix)))+1./y(ix)./beta(phi.^2.*mu(ix),y(ix)).*(phi.^2./(1+phi.^2)).^(phi.^2.*mu(ix)).*(2.*phi.*mu(ix).*log(phi.^2./(1+phi.^2))+mu(ix).*(2.*phi./(1+phi.^2)-2.*phi.^3./(1+phi.^2).^2).*(1+phi.^2))./((1+phi.^2).^y(ix))-2./beta(phi.^2.*mu(ix),y(ix)).*(phi.^2./(1+phi.^2)).^(phi.^2.*mu(ix))./((1+phi.^2).^y(ix)).*phi./(1+phi.^2);

% dpdmu(y,phi,mu) is the derivative of p(y,phi,mu) with respect to mu.
function dpdmu=dpdmu(y,phi,mu)
dpdmu=zeros(size(y));
if length(mu)==1 mu=mu*ones(size(y)); end;
ix=find(y==0);
dpdmu(ix)=(phi.^2./(1+phi.^2)).^(phi.^2.*mu(ix)).*phi.^2.*log(phi.^2./(1+phi.^2));
ix=find(y>0);
dpdmu(ix)=-1./y(ix)./beta(y(ix),phi.^2.*mu(ix)).*(phi.^2./(1+phi.^2)).^(phi.^2.*mu(ix))./((1+phi.^2).^y(ix)).*(phi.^2.*Psi(phi.^2.*mu(ix))-phi.^2.*Psi(y(ix)+phi.^2.*mu(ix)))+1./y(ix)./beta(y(ix),phi.^2.*mu(ix)).*(phi.^2./(1+phi.^2)).^(phi.^2.*mu(ix)).*phi.^2.*log(phi.^2./(1+phi.^2))./((1+phi.^2).^y(ix));