function l = expwrlp(mu,sigma,beta,y,a,b)
% EXPPWRLP log posterior for exponential power distribution.
%    l = EXPPWRLBETA(BETA,Y,MU,SIGMA,A,B) returns 
%    log p(mu, sigma, beta | y, a, b) using p(mu) = const.,
%    p(sigma) = 1/sigma, and p(1+beta) = Gamma(1+beta|a,b).

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

u = (y - mu) / sigma;

for i=1:length(beta);
  p = 2/(1+beta(i));
  c = (gamma(3/p) / gamma(1/p))^(p/2);
  logw = 0.5*gammaln(3/p) - 1.5*gammaln(1/p) - log(1+beta(i));

  l(i) = -gammaln(a) - a*log(b) + (a-1)*log(1+beta(i)) ...
      + n*logw - (n+1)*log(sigma) ...
      - (1+beta(i))/b - c*norm(u,p)^p;
end
