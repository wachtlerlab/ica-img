function l = exppwrlmu(mu,y,beta)
% EXPPWRLMU marginal log likelihood for exp. power param. mu for fixed beta.
%    The scale parameter, phi, has been integrated out. The normalizing 
%    constant, k, is not computed.

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
p = 2/(1+beta);

for i=1:length(mu)
  t = mu(i);
  M = norm(abs(y - t),p);
  l(i) = -n*log(M);
end
