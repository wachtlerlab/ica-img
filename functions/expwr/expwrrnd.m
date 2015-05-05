function r = exppwrrnd(mu,sigma,beta,m,n)
% EXPWRRND Random matrices from an exponential power distribution.
%   R = EXPWRRND(MU,SIGMA,BETA,M,N) returns an M by N matrix of random
%   numbers chosen from the exponential power distribution with
%   parameters MU, SIGMA, and BETA.  All inputs are scalar.
%
%   See also EXPWRPDF.

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

% Convert to a generalized exponential distribution

p = 2/(1 + beta);
c = (gamma(3/p) / gamma(1/p))^(p/2);
s = sigma * c^(-1/p);

r = gexprnd(s,p,m,n);
b = 2*unidrnd(2,m,n) - 3;
r = r .* b + mu;
