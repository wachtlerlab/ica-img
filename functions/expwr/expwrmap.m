function [mu,sigma,beta] = expwrmap(y,a,b,tol)
% EXPPWRMAP  Maximum aposteriori estimate of exponential power parameters.
%    [MU,SIGMA,BETA] = EXPPWRMAP(Y,A,B,TOL) returns the maximum of
%    log p(mu, sigma, beta | y, a, b) using a gamma prior on 1+beta.
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

if nargin < 4
  tol=1e-3;
end

mu = mean(y);
sigma = sqrt(var(y));

beta0 = 0;				% Gaussian

options = foptions;
options(1) = 0;			% display flag
options(2) = tol;		% soln tolerance
betamin = -0.9;
betamax = 20.0;

% unconstrained optimization is performed in space of x = log(1+beta).

xmin = log(1+betamin);
xmax = log(1+betamax);

% The estimate of beta is biased if sigma isn't marginalized
% [x,options] = fmin('expwrlp_fmin',xmin,xmax,options,mu,sigma,y,a,b);
[x,options] = fmin('expwrmlbeta_fmin',xmin,xmax,options,y,mu,a,b);

beta = exp(x) - 1;
