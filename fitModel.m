function [ Model, Result ] = fitModel (modelId, options)

% local config TODO: move to config file
shouldSaveState = 0;

% basic init
clear Model fitPar dispPar Result;

stateDir = fullfile ('..', 'state');

if exist (stateDir, 'dir') == 0
   mkdir (stateDir); 
end

start = 1;
Result.tStart = tic;

[Model, fitPar, dispPar, dataPar] = loadConfig (modelId);

Model.id = DataHash (Model, struct ('Method', 'SHA-1'));

if nargin > 1
  dispPar.plotflag = options.progress;
  fitPar.saveflag = options.savestate;
end

if dispPar.plotflag
  figure(1)
  figure(2)
  figure(3)
end

fprintf ('\nUsing GPU: %d\n', options.gpu);
fprintf ('Fitting %s for config %s [%s]\n',...
  Model.id(1:7), Model.cfgId(1:7), datestr (clock (), 'yyyymmddHHMM'));

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

  if (i == start || isUpdatePoint (i, dispPar.updateFreq, fitPar))
    Result = updateDisplay(Model, Result, fitPar, dispPar);
  end

  dA = calcDeltaA(Result.S, Model, options.gpu);
  epsilon = interpIter(i, fitPar.iterPts, fitPar.epsilon);
  epsilon = epsilon/max(abs(dA(:)));
  dA = epsilon*dA;
  Model.A = Model.A + dA;

  if (fitPar.saveflag && isUpdatePoint (i, fitPar.saveFreq, fitPar))
    saveState(Model, Result, fitPar);
  end
end

% time reporting
Result.tDuration = toc (Result.tStart);

Model.fitPar = fitPar;
Model.dispPar = dispPar;
Model.dataPar = dataPar;
Model.onGPU = options.gpu;

fprintf (['Total time: (',num2str(Result.tDuration),')\n']);

end

function [res] = isUpdatePoint (iter, freq, fitPar)
  res = (rem(iter, freq) == 0 || iter == fitPar.maxIters);
end

