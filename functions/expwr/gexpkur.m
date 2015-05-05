function k = gexpkur(s,p)
% GEXPKUR Generalized exponential kurtosis.
%    K = GEXPKUR(S,P) returns the kurtosis of the generalized exponential
%    probability density function.

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

k = gamma(1/p)*gamma(5/p)/gamma(3/p)^2;
