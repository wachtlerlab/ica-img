function hout = tilefig(n)
% tilefig -- place figure windows so that they are tiled.

dockxsize = 66;
screenxdim = 1024 - dockxsize;
screenydim = 768;
topbarsize = 50;
bottombarsize = 8;
leftbarsize = 1;
rightbarsize = 1;
spacing = 10;

window_xsize = (screenxdim - spacing)/2;
figure_xsize = window_xsize - leftbarsize - rightbarsize;

window_ysize = (screenydim - spacing)/2;
figure_ysize = window_ysize - topbarsize - bottombarsize;

h = figure(n);

if (nargout > 0)
  hout = h;
end

% For figures 1-4, tile as below, other use normal placement.
% 1 3
% 2 4

if (n == 1)
  xpos = 1 + leftbarsize;
  ypos = 1 + window_ysize + spacing + bottombarsize;
  set(h,'position',[xpos ypos figure_xsize figure_ysize]);
elseif (n == 2)
  xpos = 1 + leftbarsize;
  ypos = 1 + bottombarsize;
  set(h,'position',[xpos ypos figure_xsize figure_ysize]);
elseif (n == 3)
  xpos = 1 + window_xsize + spacing + leftbarsize;
  ypos = 1 + window_ysize + spacing + bottombarsize;
  set(h,'position',[xpos ypos figure_xsize figure_ysize]);
elseif (n == 4)
  xpos = 1 + window_xsize + spacing + leftbarsize;
  ypos = 1 + bottombarsize;
  set(h,'position',[xpos ypos figure_xsize figure_ysize]);
end
