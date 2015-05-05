function k = expwrkur(beta)
% EXPWRKUR Kurtosis of the exponential power probability density.
%    k = EXPWRKUR(BETA) returns the kurtosis of the exponential
%    power probability density which depends only on beta.

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

p = 2./(1+beta);    
k = gamma(5./p).*gamma(1./p)./gamma(3./p).^2 - 3;
