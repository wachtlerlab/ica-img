function beta = expwrmapbeta(y,mu,sigma,a,b,tol)
% EXPPWRMAP  Maximum aposteriori estimate of exponential power parameters.
%    BETA = EXPPWRMAP(Y,MU,SIGMA,A,B,TOL) returns the maximum of
%    log p(beta | y, mu, sigma, a, b) using p(1+beta) = gamma(a,b).
%    The optional argument TOL specifies the tolerance passed to FMIN.

% Written by Mike Lewicki 3/99
%
% Copyright (c) 1999 Michael S. Lewicki and CMU
%
% Permission to use, copy, modify, and distribute this software and its
% documentation for any purpose and without fee is hereby granted,
% provided that the above copyright notice and this paragraph appear in
% all copies.  Copyright holder(s) make no representation about the
% suitability of this software for any purpose. It is provided "as is"
% without express or implied warranty.

if nargin < 5
  tol=1e-3;
end

beta0 = 0;				% Gaussian

%options = foptions;
%options(1) = 0;			% display flag
%options(2) = tol;		% soln tolerance

options=optimset('Display','off','TolX',tol);

betamin = -0.9;
betamax = 20.0;

% unconstrained optimization is performed in space of x = log(1+beta).

xmin = log(1+betamin);
xmax = log(1+betamax);

%[x,options] = fmin('expwrlbeta_fmin',xmin,xmax,options,y,mu,sigma,a,b);
x = fminbnd(@expwrlbeta_fmin,xmin,xmax,options,y,mu,sigma,a,b);

beta = exp(x) - 1;
