function [ Model, Result ] = fitModel (modelId, options)

% local config TODO: move to config file
shouldSaveState = 0;

% basic init
clear Model fitPar dispPar Result;

if exist ('state', 'dir') == 0
   mkdir ('state'); 
end

start = 1;
Result.tStart = tic;

[Model, fitPar, dispPar, dataPar] = loadConfig (modelId);

if nargin > 1
  dispPar.plotflag = options.progress;
  fitPar.saveflag = options.savestate;
  
end

if dispPar.plotflag
  figure(1)
  figure(2)
  figure(3)
end

%% Prepare image data
Result.images = prepare_images (dataPar);

% Present the filtered pictures (inkluding the excluded patches)
% to the user for visual validation
if dispPar.plotflag && dataPar.doDebug
    displayImages (Result.images, dataPar, 1);
end

%% Infer matrix
%

dA = zeros (size(Model.A));
Result.priorN = 0;
Result.dataIdx = 1;
Result.X = [];		% force new dataset to be generated

for i = start : fitPar.maxIters
  Result.iter = i;

  Result = samplePats_plain(Result, fitPar, dataPar);

  if start == 1 && Result.iter == start
    Model = rescaleBfs(Model, Result);
  end

  Result.S = pinv(Model.A)*Result.D;

  [Model, Result] = adaptPrior(Model, Result, fitPar);

  if (i == start)
      Result = updateDisplay(Model, Result, fitPar, dispPar, 'init');
  elseif (rem(i, dispPar.updateFreq) == 0 || i == fitPar.maxIters)
      Result = updateDisplay(Model, Result, fitPar, dispPar);
  end

  dA = calcDeltaA(Result.S, Model);
  epsilon = interpIter(i, fitPar.iterPts, fitPar.epsilon);
  epsilon = epsilon/max(abs(dA(:)));
  dA = epsilon*dA;
  Model.A = Model.A + dA;

  if (fitPar.saveflag && (rem(i, fitPar.saveFreq) == 0 || ...
      i == fitPar.maxIters))
    saveState(Model, Result, fitPar);
  end
end

% time reporting
Result.tDuration = toc (Result.tStart);
fprintf (['Total time: (',num2str(Result.tDuration),')\n']);

end

