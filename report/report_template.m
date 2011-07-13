%% Resutls for ICA Simulation

%%
% 
%  Basis Functions
% 

start = 0;
num = 10;

patchSize = Model.patchSize;
dataDim = Model.dataDim;

[~,M] = size(Model.A);
A = sortAbf (Model.A);

hf = figure ('Name', ['Basis Functions: ', Model.name], ...
  'Position', [0, 0, 800, 1000], 'Color', 'w');

ha = tight_subplot (num, dataDim, [.001 .001], 0);

lblChan = {{'S (off)'}, {'S (on)'}, {'M (off)'}, {'M (on)'}, {'L (off)'}, {'L (on)'}};

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
    colormap ('gray')
    imagesc (shaped (:, :, n));
    cap = sprintf ('%d [%s]', idx, char (lblChan{n}));
    text (0.75, 1, cap, 'Color', 'm');
  end
  
  
end