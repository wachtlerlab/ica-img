function [ Model, Result ] = fitModel (Model, fitPar, dispPar, dataset, options)


%% Profiling
profileLen = 1000;
calcTimes = zeros(profileLen, 4);

%% Setup the Result structs
%
Result.priorN = 0;
Result.dataIdx = 1;
Result.X = [];		% force new dataset to be generated

Result.tStart = tic;

Result.S = zeros (length (Model.A), dataset.blocksize);

%% Main Loop
start = 1;

for i = start : fitPar.maxIters
  Result.iter = i;

  cT = mod(i - 1, profileLen) + 1;
  
  tstart = tic;
  Result = samplePats(Result, dataset);
  calcTimes(cT, 1) = toc(tstart);

  if start == 1 && Result.iter == start
    Model = rescaleBfs(Model, Result);
  end
  
  tstart = tic;
  
  Result.S = pinv(Model.A)*Result.D;
  
  calcTimes(cT, 2) = toc(tstart);
  tstart = tic;
  
  [Model, Result] = adaptPrior(Model, Result, fitPar);
  
  calcTimes(cT, 3) = toc(tstart);
  
  if (i == start || isUpdatePoint (i, dispPar.updateFreq, fitPar))
    Result = updateDisplay(Model, Result, fitPar, dispPar);
  end
  
  tstart = tic;
  epsilon = interpIter (i, fitPar.iterPts, fitPar.epsilon);
  
  
  [dA, A] = calcDeltaA (Result.S, Model);
  Model.A = updateAwithDeltaA (A, dA, epsilon);
  
  calcTimes(cT, 4) = toc(tstart);
  
  if (fitPar.saveflag && isUpdatePoint (i, fitPar.saveFreq, fitPar))
    tstart = tic;
    fprintf('%5d: Saving state\n',Result.iter);
    saveState(Model, Result, fitPar);
    fprintf('%5s  done in %f \n', ' ', toc(tstart));
  end
  
  if (mod (i, profileLen) == 0)
    ctm = mean(calcTimes);
    tt = sum(ctm);
    fprintf('%5s  Profile data: %f [%2.1f %2.1f %2.1f %2.1f] \n', ' ',...
      tt*profileLen, ctm(1)*100/tt, ctm(2)*100/tt, ctm(3)*100/tt, ctm(4)*100/tt);
  end
end



end

function [res] = isUpdatePoint (iter, freq, fitPar)
  res = (rem(iter, freq) == 0 || iter == fitPar.maxIters);
end


function [A] = updateAwithDeltaA (A, dA, epsilon)

  epsilon = epsilon/max(abs(dA(:)));
  dA = epsilon * dA;
  A = A + dA;

end
