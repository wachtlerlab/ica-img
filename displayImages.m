function displayImages (Result, dataPar, figNum)
%displayImages ...

set (0, 'CurrentFigure', figNum);
colormap (gray)

nx = length (Result.images);
ny = dataPar.dataDim;
ha = tight_subplot (nx, ny, [.01 .03], [.01 .01]);

n_plots = ny * nx;

refxs = [1, 3, 3, 1, 1];
refys = [2, 2, 4, 4, 2];

for idx=1:n_plots
  n_img = floor ((idx - 1) / ny) + 1;
  n_p = mod (idx  - 1, ny) + 1;
  img = Result.images(n_img).imgData;
  refkoos = Result.images(n_img).refkoos;
  bf = squeeze (img(n_p, :, :));
  set (gcf, 'CurrentAxes', ha(idx));
  %title (num2str (idx));
  hold on
  imagesc(bf');
  axis image;
  axis off;
  plot (refkoos(refxs), refkoos(refys), 'r-');
  rotate (ha(idx), [1 0 0], 180);
end

drawnow;

end

