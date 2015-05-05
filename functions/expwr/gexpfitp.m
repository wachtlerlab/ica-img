function k = gexpfitp(x,s,s_prior,p_prior)
% GEXPFITP Fit p for generalized exponential.
%    P = GEXPFITP(X,S,A,B) the MAP value of P in a generalized exponential
%    probability density function under a gen. exp prior with the parameters
%    S_PRIOR and B_PRIOR.

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

% fit by transforming x into a gamma distribution.

tol = 0.01;

a = 1/p_prior;
b = s_prior^p_prior;

init = a*b;		% expected value of p from prior

options = foptions;
options(1) = 1;                 % display flag
options(2) = tol;               % soln tolerance
pmin = 0.01;
pmax = 10.0;

[ds, options] = fmin('dslogp',dsmin,dsmax,options, ...
    d,A,x,s0,lambda,theta,logPs0,pthresh);
niters = options(10);
