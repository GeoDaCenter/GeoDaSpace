% This is the Matlab script file to analyze the grub data. Please use
% Matlab 6.5 or earlier. Matlab 7.0 and later versions contain a confirmed
% bug in fminunc().
%
% Data is in text file LocOMGrubs.txt, assumed to be stored in the same 
% directory as this m-file.
%
% Functions called:
% NegLogEL() (see NegLogEL.m)
%
% Required Matlab toolboxes:
% Optimization
% Statistics
%
% Required library:
% specfun (by Paul Godfrey, available at my.fit.edu/~gabdo/paulbio.html)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

data=load('LocOMGrubs.txt'); % Read data file. Columns 1 and 2 are xy coordinates; Column 3 is organic matter; Column 4 is grub count.
xy=data(:,1:2);
n=length(xy);
%Create distance matrix H.
Zx=xy(:,1)*ones(1,n);
Zy=xy(:,2)*ones(1,n); 
H=sqrt((Zx-Zx').^2+(Zy-Zy').^2);

% Set bounds for some of the parameters to ensure numerical stability.
B=-log(.05)/(min(min(H(H>0)))); %Set upper bound for decay parameter so that effective range is no smaller than min dist between points.
Bphi=100; %Set upper bound for phi.

%Create design matrix XX and calculate (XX'XX)^(-1).
XX=[ones(n,1) data(:,3) data(:,3).^2 data(:,3).^3];
XXpXXInv=(XX'*XX)^-1;

%Set initial values for parameters.
alphaN0=.5;
alphaR0=-log(.1/alphaN0)/min(min(H(H>0)));

beta0=glmfit(XX(:,2:4),data(:,4),'poisson');
y=data(:,4);
muY=exp(XX*beta0);
vy=(y-muY).^2;

phi0=4.3478; % This one gives (1+phi0)/phi0=1.23, which is TPL intercept.
% Other possible starting values for phi (use these if algorithm doesn't
% converge):
%phi0=min(abs((muY./(vy-muY))));
%phi0=mean(abs((muY./(vy-muY))));
%phi0=max(abs((muY./(vy-muY))));


% Set initial parameter values. Constrained parameters are transformed
% within NegLogEL fminunc can be used. Therefore, reverse-transform them
% here.
theta0=[log(alphaN0/(1-alphaN0)) log((alphaR0/B)/(1-alphaR0/B)) log((phi0/Bphi)/(1-phi0/Bphi)) beta0'];
dimbeta=length(beta0);

% Generate random uniforms to accomplish jittering.
numu=10000;
%U=unifrnd(0,1,length(y),numu);
% Load saved generated values
U = importdata('U.txt')

% Set optimization options and call fminunc.
options=optimset('GradObj','on','LargeScale','off','DerivativeCheck','off','Display','iter');
warning off MATLAB:divideByZero
[theta,fval,exitflag,output,grad,hessian]=fminunc(@NegLogEL,theta0,options,y,U,XX,H,dimbeta,B,Bphi)

% Invert Hessian to get estimated var(thetahat).
Hinv=hessian^(-1);

% Print out thetahat and standard error.
theta
SE=sqrt(diag(Hinv))'
