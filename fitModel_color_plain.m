function [Model, Result] = fitModel_color_plain(Model, fitPar, dispPar)

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

[L,M] = size(Model.A);
dA = zeros(size(Model.A));
Result.priorN = 0;
Result.dataIdx = 1;
Result.X = [];		% force new dataset to be generated

for i = start : fitPar.maxIters
  Result.iter = i;

  Result = samplePats_plain(Result, fitPar);

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
