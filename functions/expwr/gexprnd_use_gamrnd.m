function r = gexprnd(s,p,m,n)
% EXPRND Random matrices from a generalized exponential distribution.
%   R = GEXPRND(S,P,M,N) returns an M by N matrix of random numbers chosen
%   from the generalized exponential distribution with parameters S and P.
%   All parameters are scalar.
%
%   See also GEXPPDF.

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

a = 1./p;
b = s.^p;

l = m*n;

maxlen=200;

if (l > maxlen)
  fprintf('splitting return matrix size due to matlab bug..\n');
  
  r = zeros(l,1);
  for i=1:maxlen:l
    fprintf('\r%d-%d',i,i+maxlen-1);
    % fprintf('\n   gamrnd(%.3f,%.3f,%d,%d);\n',a,b,maxlen,1);
    r(i:i+maxlen-1) = gamrnd(a,b,maxlen,1);
  end
  r = reshape(r,m,n);
  fprintf('\n');
else
  r = gamrnd(a,b,m,n);
end

r = r.^(1/p);
