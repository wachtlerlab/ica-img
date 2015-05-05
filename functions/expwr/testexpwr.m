% params for expwr
mu = 0;
sigma = 1;
beta = 2;

fprintf('ExPwr: mu=%5.2f  sigma=%5.2f  beta=%+5.2f\n', mu,sigma,beta);

% equivalent params for gexp
p = 2/(1 + beta);
c = (gamma(3/p) / gamma(1/p))^(p/2);
s = sigma * c^(-1/p);

fprintf('GExp: s=%5.2f  p=%5.2f\n',s,p);

n = 1000;
l = 10*sigma;
dx = 2*l/n;
x = -l:dx:l;
hx = x + dx/2;

y = expwrpdf(x,mu,sigma,beta);

figure(1);
plot(x,y);

z = sum(y*dx);
k = expwrkur(beta);
fprintf('z=%5.3f  k=%5.3f\n',z,k);

npoints = 500;
nbins = 100;
l = 10*sigma;
dx = 2*l/nbins;
x = -l:dx:l;
hx = x + dx/2;

r = expwrrnd(mu,sigma,beta,npoints,1);

figure(2);
hist(r,nbins);

t= -0.9:0.05:10;
a = 2;
b = 2;
l = gampdf(t+1,a,b);
figure(3);
plot(t,l);

% l = expwrlbeta(t,r,mu,sigma,a,b);
l = expwrmlbeta(t,r,mu,a,b);
figure(4);
d = log(1+t);
h = plot(d,l);

ax = gca;
xinc = (max(t) - min(t))/5;
btick = [min(t):xinc:max(t)];
dtick = log(1 + btick);
set(ax, 'XTick', dtick, 'XTickLabels',btick);

[mu_mp, sigma_mp, beta_mp]  = expwrmap(r,a,b);

fprintf('ExPwr map: mu=%5.2f  sigma=%5.2f  beta=%+5.2f\n', ...
    mu_mp, sigma_mp, beta_mp);
