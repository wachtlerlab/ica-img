function y = gexppdf(x,s,p)
% GEXPPDF Generalized exponential probability density function.
%    Y = GEXPPDF(X,S,P) returns the generalized exponential probability
%    density function:
%
%       y = c * exp(-(x/s)^p)
%
%    where c = p / (s * gamma(1/p));
%    y is (half) Gaussian for p = 2 and exponential for p = 1.
%    y approaches a uniform distribution as p approaches infinity.

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

c = p / (s * gamma(1/p));
y = c * exp(-(x/s).^p);
