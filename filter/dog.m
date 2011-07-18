function [ y ] = dog (x, mu, s_i, s_o)
%DOG Differance of Gaussions

y = do_gauss (x, mu, s_i) - do_gauss (x, mu, s_o);

end

function [out] = do_gauss (x, mu, sigma)

out = 1/(2*pi*sigma^2)*exp(-(x - mu).^2/(2*sigma^2));

end