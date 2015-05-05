function l = expwrlbeta_fmin(x,y,mu,sigma,a,b)

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

% to do unconstrained optimization of beta \in [-1,inf],
% we optimize in the space of x = log(1 + beta)
beta =  exp(x) - 1;

% we're minimizing, so return negative
l = -expwrlbeta(beta,y,mu,sigma,a,b);
