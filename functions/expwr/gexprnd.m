function r = gexprnd(s,p,m,n)
% EXPRND Random matrices from a generalized exponential distribution.
%   R = GEXPRND(S,P,M,N) returns an M by N matrix of random numbers chosen
%   from the generalized exponential distribution with parameters S and P.
%   All parameters are scalar.
%
%   See also GEXPPDF.

% Copyright (c) 1999 Michael S. Lewicki and CMU
%
% Permission to use, copy, modify, and distribute this software and its
% documentation for any purpose and without fee is hereby granted,
% provided that the above copyright notice and this paragraph appear in
% all copies.  Copyright holder(s) make no representation about the
% suitability of this software for any purpose. It is provided "as is"
% without express or implied warranty.

if (p == 1)
  % there seems to be a bug in gamrnd for this case where the
  % returned distribution has twice the variance
  r = exprnd(s,m,n);
else
  a = 1/p;
  b = s^p;
  r = gamrnd2(a,b,m,n);
  r = r.^(1/p);
end
