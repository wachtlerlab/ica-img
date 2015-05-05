function y = expwrpdf(x,mu,sigma,beta)
% EXPWRPDF Exponential power probability density function.
%    Y = EXPWRPDF(X,THETA,PHI,BETA) returns the exponential power probability
%    density function:
%
%       y = p(x|mu,sigma,beta)
%         = (w(beta)/sigma) * exp[-c(beta)*|(x - mu)/sigma)|^(2/(1 + beta))]
%
%    -inf < theta < inf, phi > 0, beta > -1
%    Letting p = 2/(1 + beta), c = (gamma(3/p) / gamma(1/p))^(p/2), and
%    w = (gamma(3/p))^0.5 / ((2/p) * (gamma(1/p))^1.5).
%    E[x] = mu, Var[x] = sigma.
%
%    y is Gaussian for beta = 0, uniform for beta = -1, and double
%    exponential for beta = 1.  See Box and Tiao "Bayesian Inference in
%    Statistical Analysis" (1973)

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

p = 2/(1 + beta);
w = (gamma(3/p))^0.5 / ((2/p) * (gamma(1/p))^1.5);
c = (gamma(3/p) / gamma(1/p))^(p/2);

y = (w/sigma) * exp(-c*abs((x - mu)./sigma).^p);
