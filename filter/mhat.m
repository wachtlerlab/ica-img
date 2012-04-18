function [ Z ] = mhat (nvals, si, so, r, mu)

if nargin<4
  mu=0;
end

n = floor (nvals/2.0);

[X, Y] = meshgrid(-n:n);
[~, R] = cart2pol (X, Y);
Z = dog (R, mu, si, so, r);
Z = Z/max(abs(Z(:)));

end

