function [Model, Result] = fitModel_color_plain(Model, fitPar, dispPar, DataParam)

% Written by Mike Lewicki 4/99
%
% Copyright (c) 1999 Michael S. Lewicki and CMU
%
% Permission to use, copy, modify, and distribute this software and its
% documentation for any purpose and without fee is hereby granted,
% provided that the above copyright notice and this paragraph appear in
% all copies.  Copyright holder(s) make no representation about the
% suitability of this software for any purpose. It is provided "as is"
% without express or implied warranty.

if (fitPar.startIter > 1)
  stateFile = sprintf('%s-state-rev-i%d', Model.name, fitPar.startIter);
  [Model, Result] = loadState(stateFile);
  start = fitPar.startIter + 1;		% start at iter after saved state
else
  start = 1;
end

Result.images = prepare_images (DataParam);

% Present the filtered pictures (inkluding the excluded patches)
% to the user for visual validation

if DataParam.doDebug
    figure (1);
    set (0, 'CurrentFigure', 1); 
    colormap (gray)
    
    nx = length (Result.images);
    ny = DataParam.dataDim;
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

[L,M] = size(Model.A);
dA = zeros(size(Model.A));
Result.priorN = 0;
Result.dataIdx = 1;
Result.X = [];		% force new dataset to be generated

for i = start : fitPar.maxIters
  Result.iter = i;

  Result = samplePats_plain(Result, fitPar, DataParam);

  if start == 1 & Result.iter == start
    Model = rescaleBfs(Model, Result);
  end

  Result.S = pinv(Model.A)*Result.D;

  [Model, Result] = adaptPrior(Model, Result, fitPar);

  if (i == start)
      Result = updateDisplay_color(Model, Result, fitPar, dispPar, 'init');
  elseif (rem(i, dispPar.updateFreq) == 0 | i == fitPar.maxIters)
      Result = updateDisplay_color(Model, Result, fitPar, dispPar);
  end

  dA = calcDeltaA(Result.S, Model);
  epsilon = interpIter(i, fitPar.iterPts, fitPar.epsilon);
  epsilon = epsilon/max(abs(dA(:)));
  dA = epsilon*dA;
  Model.A = Model.A + dA;

  if (rem(i, fitPar.saveFreq) == 0 | i == fitPar.maxIters)
    saveState(Model, Result, fitPar);
  end
end
