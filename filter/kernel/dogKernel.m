function [ kernel ] = dogKernel (cfg)

nvals = cfg.size;
si = cfg.s_inner;
so = cfg.s_outer;
r = cfg.ratio_io;

if isfield (cfg, 'mu')
  mu = cfg.mu;
else
  mu = 0;
end

n = floor (nvals/2.0);

[X, Y] = meshgrid(-n:n);
[~, R] = cart2pol (X, Y);
Z = dog (R, mu, si, so, r);

kernel = Z/max(abs(Z(:)));

end
