function dA = calcdA_expwr(X,A,S,mu,sigma,beta)
% calcdA_expwr -- calc d/dA P(X|A) using P(s_m) = ExPwr(s_m | mu,sigma,beta)
%   Usage
%     dA = calcdA_expwr(X,A,S,mu,sigma,beta)
%   Inputs
%      X         data vectors
%      A         basis functions
%      S         most probable coefficients for each pattern
%      mu        vector of means for each prior P(s_m)
%      sigma     vector of standard deviations for each prior P(s_m)
%      beta      vector of kurtosis parameters for each prior P(s_m)
%   Outputs
%      dA        gradient of A

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

npats = size(X,2);
[L,M] = size(A);
Z = zeros(size(S));

dA = zeros(size(A));

for m=1:M
  s = S(m,:) - mu(m);
  q = 2/(1+beta(m));
  c = (gamma(3/q)/gamma(1/q))^(q/2);
  Z(m,:) = -(q*c/(sigma(m)^q)) * abs(s).^(q-1) .* sign(s);
end

dA = -A*Z*S' - npats*A;

% normalize by the number of patterns
dA = dA/npats;
