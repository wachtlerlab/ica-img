function y = coshpdf(x,theta,beta)
% COSHPDF Hyperbolic cosine probability density function.
%    Y = COSTPDF(X,THETA,BETA) returns the density function:
%
%       y = p(x|theta,beta)
%         = theta * ( cosh(beta * x) )^(-theta/beta);
%
%       theta, beta > 0

% Written by Mike Lewicki 9/99
%
% Copyright (c) 1999 Michael S. Lewicki and CMU
%
% Permission to use, copy, modify, and distribute this software and its
% documentation for any purpose and without fee is hereby granted,
% provided that the above copyright notice and this paragraph appear in
% all copies.  Copyright holder(s) make no representation about the
% suitability of this software for any purpose. It is provided "as is"
% without express or implied warranty.

% (z courtesy of Mathematica :-)
z = 2*sqrt(pi) * gamma(1 + 0.5*theta/beta) / gamma( 0.5*(theta+beta) / beta );

y = theta * ( cosh(beta * x) ).^(-theta/beta)  / z;
