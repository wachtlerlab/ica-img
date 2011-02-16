function logL = calcLogPXA(X,S,Model)
% calcLogPXA -- log likelihood using the low noise limit assuming expwr prior

% Written by Mike Lewicki 4/99
%
% Copyright (c) 1999 Michael S. Lewicki and CMU
%
% Permission to use, copy, modify, and distribute this software and its
% documentation for any purpose and without fee is hereby granted,
% provided that the above copyright notice and this paragraph appear in
% all copies.  Copyright holder(s) make no representation about the
% suitability of this software for any purpose. It is provided "as is"
% without express or implied warranty.

N = size(X,2);		% number of patterns
A = Model.A;
[L,M] = size(A);
mp=Model.prior;

% for a single pattern x, 
%    log P(x|A) =  log P(S) - log |det A|

logPS = 0;
for m=1:M
  logPS = logPS + sum(expwrpdfln(S(m,:),mp.mu(m),mp.sigma(m),mp.beta(m)));
end
logL = logPS - N*log(abs(det(A)));
