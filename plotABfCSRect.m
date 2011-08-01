function [hf] = plotABfCSRect (Model, start, num, figHandle)

if (nargin < 4)
  hf = figure ('Name', ['Basis Functions: ', Model.name], ...
   'Position', [0, 0, 800, 1000], 'Color', 'w', 'PaperType', 'A4');
else
  hf = figHandle;
end

if (nargin < 3)
  num = 10;
end;

if (nargin < 2)
  start = 0;
else
  start = start - 1;
end

patchSize = Model.patchSize;
dataDim = Model.dataDim;

[~,M] = size(Model.A);
A = sortAbf (Model.A);
A = normABf (A);
ha = tight_subplot (num, dataDim, [.01 .01], 0);

lblChan = {{'S (on)'}, {'S (off)'}, {'M (on)'}, {'M (off)'}, {'L (on)'}, {'L (off)'}};

for ii = 1:num
  
  idx = start + ii;
  
  bf = A (:, idx);
  R = reshape (bf, dataDim, patchSize, patchSize);
  shaped = permute (R, [3 2 1]); % x, y, channel
 
  for n = 1:dataDim
    curAxis = (ii - 1) * dataDim + n;
    set (gcf, 'CurrentAxes', ha(curAxis));
    hold on;
    %%axis image;
    axis off;
    axis equal;
    colormap ('gray')
    image (shaped (:, :, n)*100);
    cap = sprintf ('%d [%s]', idx, char (lblChan{n}));
    text (0.75, 1, cap, 'Color', 'm');
  end
  
end

end