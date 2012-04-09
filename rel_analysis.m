function rel_analysis(Model)

chmap{1} = 'S+';
chmap{2} = 'S-';
chmap{3} = 'L+';
chmap{4} = 'L-';

[rows, cols] = size (Model.A);

nchan = 4;

A = sortAbf(Model.A);
n = cols/nchan;
X = reshape(A, 4, rows * n);

combs = combnk(1:nchan,2);

fig = figure();

ax = {};

for c = 1:size (combs, 1)
  
  ax{c} = subplot(2,3,c);
  
  x = X(combs(c, 1), :);
  y = X(combs(c, 2), :);
  
  scatter (x, y, 'k.')
  
  xlabel(chmap{combs(c, 1)})
  ylabel(chmap{combs(c, 2)})
   
end

[Xica, A, W] = fastica (X);

colors = lines(nchan);

for c = 1:size (combs, 1)
 
  
 for n = 1:4
   l = A([combs(c, 1), combs(c, 2)], n);
   plot_direction (l, ax{c}, colors(n, :))
 end
  
end

end



function plot_direction(l, ax, color)

m = l(2)/ l(1);

x = [-0.5, 0.5];
p = [m, 0];

y = polyval(p, x);

set (gcf, 'CurrentAxes', ax);
line (x, y, 'Color', color)
xlim ([-1, 1])
ylim ([-1, 1])

end

