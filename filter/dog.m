function [ y ] = dog (x, mu, s_i, s_o, ratio)
%DOG Differance of Gaussians
%   (c) Christian Kellner <kellner@bio.lmu.de>
%       License: BSD-3 clause

if nargin < 5
  ratio = 1.0;
end

y = do_gauss (x, mu, s_i) - ratio * do_gauss (x, mu, s_o);

end

function [out] = do_gauss (x, mu, sigma)
out = 1/(2*pi*sigma^2)*exp(-(x - mu).^2/(2*sigma^2));
end