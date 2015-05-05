function logy = expwrpdfln(x,mu,sigma,beta)
% EXPWRPDF Exponential power log probability density function.
%    LY = EXPWRPDFLN(X,MU,SIGMA,BETA) returns the logarithm of the
%    exponential power probability density function.
%    See EXPWRPDF

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

logw = 0.5*gammaln(1.5*(1+beta)) - log(1+beta) - 1.5*gammaln(0.5*(1+beta));
c = (gamma(1.5*(1+beta)) / gamma(0.5*(1+beta)))^(1/(1+beta));
logy = logw - log(sigma) - c*abs((x - mu)./sigma).^(2/(1+beta));
