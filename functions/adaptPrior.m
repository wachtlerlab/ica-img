function [Model, Result] = adaptPrior(Model, Result)

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

adaptSize = Model.prior.adaptSize;

if adaptSize == 0		% 0 means don't adapt
  return
end

[L,M] = size(Model.A);
[L,N] = size(Result.S);
mp = Model.prior;

if Result.priorN == 0
  Result.priorS = zeros(M,adaptSize);
  Result.priorIdx = 1;
end

% add current coeffs to dataset for prior

N = min( N, adaptSize );

a = Result.priorIdx;
if a+N-1 <= adaptSize
  b = Result.priorIdx + N - 1;
  Result.priorS(:,a:b) = Result.S(:,1:N);
else
  % handle wrap around
  b = adaptSize;
  n = b-a+1;
  Result.priorS(:,a:b) = Result.S(:,1:n);
  a = 1;
  b = N - n;
  Result.priorS(:,a:b) = Result.S(:,n+1:N);
end

% adapt every time the buffer is filled

if Result.priorIdx + N >= adaptSize
  fprintf('%5d: Updating prior\n',Result.iter);
  
  for m=1:M;
    mp.beta(m) = expwrmapbeta(Result.priorS(m,:), mp.mu(m), mp.sigma(m), ...
      mp.a, mp.b, mp.tol);
  end
  Model.prior = mp;
  
end

Result.priorIdx = mod(Result.priorIdx + N, adaptSize);
Result.priorN   = min(Result.priorN + N, adaptSize);
