function [ pcs ] = plotDiDirs(model)

A = model.A;

chmap = arrayfun(@(x) chan2str(x), mapChannel (model.cfg.data.filter.center, 1), 'UniformOutput', 0);
ps = model.cfg.data.patchsize;

N = 4;
A = sortAbf(A);
[~, n] = size (A);
B = reshape (A, N, ps^2, n);

pcs = zeros (N, N, n);
lts = zeros (N, n);
for k = 1:n
  X = B(:,:,k);
  [pc,~,latent,~] = princomp(X');
  %pc = pc(:,1) * latent (1);
  pcs(:,:,k) = pc;
  lts(:,k) = latent;
  %disp (pc);
end

mpc = squeeze (pcs(:,1,:))';

X = reshape(A, 4, ps^2 * n);
combs = combnk(1:N,2);
fig = figure();

ax = {};
colors = lines(size (combs, 1));

lts = lts/max(lts(:));

for c = 1:size (combs, 1)
  
  ax{c} = subplot(2,3,c);
  
  x = X(combs(c, 1), :);
  y = X(combs(c, 2), :);
  
  
  scatter (x, y, 10, 'k+');
  hold on;
  
  xlabel(chmap{combs(c, 1)})
  ylabel(chmap{combs(c, 2)})
  
  for k = 1:n
    l = pcs([combs(c, 1), combs(c, 2)], 1, k);
    %l = pc([combs(c, 1), combs(c, 2)], :);
    plot_direction (l, ax{c}, colors(c, :), lts(1, k))  
  end
  
end

end


function plot_direction(l, ax, color, len)

if nargin < 4
  len = 1.0;
end

m = l(2)/ l(1);

x = [-0.5, 0.5];
p = [m, 0];
y = polyval(p, x);

[t, r] = cart2pol (x, y);
r = [1.0, 1.0] * len;
[x, y] = pol2cart (t, r);

set (gcf, 'CurrentAxes', ax);
hold on;
line (x, y, 'Color', color);
xlim ([-1, 1])
ylim ([-1, 1])

end

