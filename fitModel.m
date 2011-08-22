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


fprintf ('\nFitting %s for config %s [%s]\n',...
  Model.id(1:7), Model.cfgId(1:7), datestr (clock (), 'yyyymmddHHMM'));

%% Setup GPU context
fprintf ('\nUsing GPU: %d\n', options.gpu);
if options.gpu
  Model.gpu = gpuDevice;
  gpuContext.absmax = absmax_setup (Model.A);
  gpuContext.calc_z = calc_z_setup (Model.A, fitPar.blocksize);
else
  gpuContext = 0;
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

  if (i == start || isUpdatePoint (i, dispPar.updateFreq, fitPar))
    Result = updateDisplay(Model, Result, fitPar, dispPar);
  end
  
  epsilon = interpIter (i, fitPar.iterPts, fitPar.epsilon);
  [dA, A] = calcDeltaA (Result.S, Model, gpuContext);
  Model.A = updateAwithDeltaA (A, dA, epsilon, gpuContext);

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


function [A] = updateAwithDeltaA (A, dA, epsilon, gpuContext)

if isstruct (gpuContext)
  e = gpuArray (epsilon);
  e = e / absmax_cu (dA, gpuContext);
  dA = e * dA;
  A = A + dA;
  A = gather (A);
else
  epsilon = epsilon/max(abs(dA(:)));
  dA = epsilon * dA;
  A = A + dA;
end




end

