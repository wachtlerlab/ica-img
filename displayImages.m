function displayImages (images, dataPar, figNum)
%displayImages ...

if nargin < 3
  figNum = figure;
else
  set (0, 'CurrentFigure', figNum);
end

set (figNum, 'Color', [1 1 1]);

colormap (gray)

nx = length (images);
ny = dataPar.dataDim;
ha = tight_subplot (nx, ny, [.01 .03], [.01 .01]);

n_plots = ny * nx;

refxs = [1, 3, 3, 1, 1];
refys = [2, 2, 4, 4, 2];
lblChan = {{'S (on)'}, {'S (off)'}, {'M (on)'}, {'M (off)'}, {'L (on)'}, {'L (off)'}};


for idx=1:n_plots
  n_img = floor ((idx - 1) / ny) + 1;
  n_p = mod (idx  - 1, ny) + 1;
  img = images(n_img).imgData;
  refkoos = images(n_img).refkoos;
  bf = squeeze (img(n_p, :, :));
  set (gcf, 'CurrentAxes', ha(idx));
  hold on
  fi = imagesc(bf');
  axis image;
  axis off;
  fk = plot (refkoos(refxs), refkoos(refys), 'r-');
  cap = sprintf ('%s', char (lblChan{n_p}));
  text (5, 15, cap, 'Color', 'r');
  rotate (fi, [0 0 1], 180);
  rotate (fk, [0 0 1], 180);
end

drawnow;

end

