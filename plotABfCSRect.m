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

Model = sortModelA (Model);
[~,M] = size(Model.A);
A = Model.A;
A = normABf (A);
ha = tight_subplot (num, dataDim*2, [.001 .02], 0);

if dataDim == 6
  lblChan = {{'S|on'}, {'S|off'}, {'M|on'}, {'M|off'}, {'L|on'}, {'L|off'}};
else
  lblChan = {{'S'}, {'M'}, {'L'}};
end


wp = ones (M,M) * 255;
scaleImg = 100;

for ii = 1:num
  
  idx = start + ii;
  
  bf = A (:, idx);
  R = reshape (bf, dataDim, patchSize, patchSize);
  shaped = permute (R, [3 2 1]); % x, y, channel
  
  for n = 1:dataDim
    curAxis = (ii - 1) * (dataDim*2) + ((n * 2));
    set (gcf, 'CurrentAxes', ha(curAxis));
    hold on;
    %%axis image;
    axis off;
    axis equal;
    colormap ('gray')
    image (shaped (:, :, n)*scaleImg);
    
    curAxis = (ii - 1) * (dataDim*2) + ((n * 2) - 1);
    set (gcf, 'CurrentAxes', ha(curAxis));
    hold on;
    axis equal;
    %%axis image;
    axis off;

    colormap ('gray')
    image (wp);
    cap = sprintf ('%3d %s', idx, char (lblChan{n}));
    text (1, 0.8*M, cap, 'Color', 'black', 'FontSize', 8);
    
    xcr = mod(n,dataDim)+1;
    cr = do_corr (R(n,:), R (xcr,:));
    
    xcr2 = mod(n+1,dataDim)+1;
    cr2 =  do_corr (R(n,:), R (xcr2,:));
    
    txt = sprintf ('%.2f %s', cr, char (lblChan{xcr}));
    text (1, (M/2)+10, txt, 'Color', 'red', 'FontSize', 5);
    txt = sprintf ('%.2f %s', cr2, char (lblChan{xcr2}));
    text (1, (0.2*M), txt, 'Color', 'blue', 'FontSize', 5);
    
  end
  
end
end


function [out] = do_corr (a, b)

out = sum (a.*b);
%out = max (xcorr (a, b, 'coeff'));
end