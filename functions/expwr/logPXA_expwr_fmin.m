function f = logPXA_expwr_fmin(vecA, Prob, X, Model, fitPar)

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

Model.A = reshape(vecA,size(Model.A));;

% get new sample of patterns
r = ceil(size(X,2)*rand(fitPar.blocksize,1));
D = X(:,r);
N = size(D,2);

% compute coefs
S = pinv(Model.A)*D;
logPXA = calcLogPXA(D,S,Model);

% normalize by number of patterns
f = -logPXA/N;
