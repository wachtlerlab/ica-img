function logE = logPXA_expwr(X,A,S,mu,sigma,beta)
% logPXA -- log likelihood using the low noise limit assuming expwr prior
%   Usage
%     logE = logPXA_expwr(X,A,mu,sigma,beta)
%   Inputs
%      X       data vector
%      A       basis functions
%      S       solution vector
%      mu      vector of means for each prior P(s_m)
%      sigma   vector of standard deviations for each prior P(s_m)
%      beta    vector of kurtosis parameters for each prior P(s_m)
%   Outputs
%      logE    Gaussian approximation to P(X|A,S,lambda,theta)

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
[L,M] = size(A);

if (L ~= M) 
  error('A must be square.\n');
end

if det(A) == 0.0
  fprintf('det(A) is zero.\n');
  return
end
 
% for a single pattern x, 
%    log P(X|A) =  log P(S) - log |det A|

logPS = 0;
for m=1:M
  s = S(m,:);
  logPS = logPS + sum(expwrpdfln(s,mu(m),sigma(m),beta(m)));
end

logdetA = log(abs(det(A)));

logE = logPS - N*logdetA;
