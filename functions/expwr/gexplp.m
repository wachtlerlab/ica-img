function l = gexplnp(pv,y,s,a,b)
% GEXPLNP marginal log likelihood for generalized exponential parameter p.
%    l = GEXPLNP(P,Y,S,A,B) returns log p(p | y, s, a, b) using a gamma
%    prior for p(p).

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

for i=1:length(pv);
  p = pv(i);
  l(i) = -gammaln(a) - a*log(b) - (a-1)*log(p) ...
      + n*(log(p) - log(2) - gammaln(1/p) - log(s)) ...
      - p/b - norm(y,p)^p;
end
