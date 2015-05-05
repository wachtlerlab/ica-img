function l = expwrmlbeta(beta,y,mu,a,b)
% EXPPWRMLBETA marginal log likelihood for exp. power parameter beta.
%    l = EXPPWRLBETA(BETA,Y,MU,A,B) returns log p(beta | y, mu, a, b),
%    where sigma has been integrated out assuming p(sigma) = 1/sigma.
%    Y is the data vector and MU is the mean of the distribution.
%    The parameters a and b define a gamma prior on 1+beta.

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

n = length(y);

for i=1:length(beta);
  p = 2/(1+beta(i));
  logpb = (a-1)*log(1 + beta(i)) - (1 + beta(i))/b;
  logM = p*log(norm(y - mu,p));

  l(i) = logpb - (n/p)*logM + gammaln(1 + n/p) - n*gammaln(1 + 1/p);
end
