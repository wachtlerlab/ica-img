function checkGradient (cfg, fig)

max_iter = (cfg.data.ncluster * cfg.data.npats / cfg.data.blocksize);
gradient = createGradient(cfg.gradient, max_iter);

iter = 1:max_iter;
points = zeros (length(iter), 1);
for n = 1:length(iter)
  points(n) = gradientGetEpsilon (gradient, iter(n));
end

if nargin < 2
  figure ('Name', ['cfg: ' cfg.id(1:7)]);
else
  figure(fig);
end

plot (iter, points, 'r');

end

