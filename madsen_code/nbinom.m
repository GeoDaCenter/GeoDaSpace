function p=nbinom(y,phi,mu)
p=zeros(size(y));
if length(mu)==1 mu=mu*ones(size(y)); end;
ix=find(y==0);
p(ix)=(phi.^2./(1+phi.^2)).^(phi.^2.*mu(ix));
ix=find(y>0);
p(ix)=1./(y(ix).*beta(y(ix),phi.^2.*mu(ix))).*(phi.^2/(1+phi.^2)).^(phi.^2.*mu(ix)).*(1./(1+phi.^2).^y(ix));


