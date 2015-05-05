function displayImages (imageset, figNum)
%displayImages ...

if nargin < 2
  figNum = figure;
else
  set (0, 'CurrentFigure', figNum);
end

set (figNum, 'Color', [1 1 1]);

colormap (gray)

images = imageset.images;
dataDim = imageset.shape(1);

nx = length (images);
ny = dataDim + 2;
ha = tight_subplot (nx, ny, [.01 .03], [.01 .01]);

refxs = [1, 3, 3, 1, 1];
refys = [2, 2, 4, 4, 2];

chanMap = imageset.channels;

for idx=1:nx
  fprintf ('\nImage %s\n', images{idx}.filename);
  curImg = images{idx};
  img = getImgData (curImg, 0);
  d = img(:);
  fprintf ('min: %.5f max: %.5f std: %.5f\n', ...
    min (d), max (d), std (d));
  img = getImgData (curImg, 0);
  d = img(:);
  fprintf ('min: %.5f max: %.5f std: %.5f\n', ...
    min (d), max (d), std (d));

  if isfield(images{idx}, 'refkoos')
      refkoos = images{idx}.refkoos;
  end

  for n = 1:(ny - 2)
      
    cur_plot = (idx - 1) * ny + n;

    chan = squeeze (img(n, :, :));
    set (gcf, 'CurrentAxes', ha(cur_plot));
    hold on
    fi = imagesc(chan');
    axis image;
    axis off;
    
    if exist('refkoos', 'var')
        fk = plot (refkoos(refxs), refkoos(refys), 'r-');
    end
    
    
    n_img = n;
    
    fprintf ('\t [%s] min: %.5f max: %.5f std: %.5f\n', ...
      chan2str (n_img, chanMap),...
      min(chan(:)), max(chan(:)), std(chan(:)));
    
    text (5, 15, chan2str (n_img, chanMap), 'Color', 'r');
    rotate (fi, [0 0 1], 180);
    if exist('refkoos', 'var')
        rotate (fk, [0 0 1], 180);
    end
    
  end
  
  if dataDim == 6
    set (gcf, 'CurrentAxes', ha((idx - 1) * ny + (ny - 1)));
    d = getImgData (curImg, 1, [5 3 1]);
    p = permute (d, [3 2 1]);
    image (p);
    axis off;
  
    set (gcf, 'CurrentAxes', ha((idx - 1) * ny + ny));
    d = getImgData (curImg, 1, [6 4 2]);
    p = permute (d, [3 2 1]);
    image (p);
    axis off;
  elseif dataDim == 4
    s = size (curImg);
    set (gcf, 'CurrentAxes', ha((idx - 1) * ny + (ny - 1)));
    d = getImgData (curImg, 1, [3 1]);
    d(3,:,:) = zeros (s(2), s(2));
    p = permute (d, [3 2 1]); 
    imagesc (p);
    axis off;
    
    set (gcf, 'CurrentAxes', ha((idx - 1) * ny + ny));
    d = getImgData (curImg, 1, [4 2]);
    d(3,:,:) = zeros (s(2), s(2));
    p = permute (d, [3 2 1]);
    imagesc (p);
    axis off;
    
  elseif dataDim == 3
    s = size (curImg);
    set (gcf, 'CurrentAxes', ha((idx - 1) * ny + ny));
    d = getImgData (curImg, 1);
    p = permute (d, [3 2 1]);
    %p(3,:,:) = zeros (s(2), s(2));
    imagesc (p);
    axis off;
    
  end
end

drawnow;

end

function [data] = getImgData (img, doNorm, channels)
data = img.data;

if doNorm
  %C = num2cell (data, 1);
  %C = cellfun (@normA, C, 'UniformOutput', false);
  %data = cell2mat (C);
  
  %data = 0.5 + 0.5 * (data / max(abs(data(:))));
  data = normA(data);
  if nargin > 2
    data = data(channels, :,:);
  end
  
end

end


function [normed] = normA (A)
A = A - min (A(:));
normed =  A / max(abs(A(:)));
%normed = normed * 255;
end

function [out] = selectChannel (Abf, cols, l, dataDim)

shaped = reshape (Abf, dataDim, l*l);
X = shaped(cols, :); 
out = reshape (X, length(cols)*l, 1);

end

