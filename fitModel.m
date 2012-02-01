function [ Model, Result ] = fitModel (Model, fitPar, dispPar, dataset, Result, options)

%% Setup GPU context
fprintf ('\nUsing GPU: %d\n', options.gpu);
if options.gpu
  %Model.gpu = gpuDevice;
  %fprintf ('\t[%s]\n', Model.gpu.Name);
  %gpuContext.absmax = absmax_setup (Model.A);
  %gpuContext.calc_z = calc_z_setup (Model.A, fitPar.blocksize);
  hcube = cube()
  hcube.setup()
  gpuContext = 0;
else
  gpuContext = 0;
  hcube = 0;
end

%% Profiling
profileLen = 1000;
calcTimes = zeros(profileLen, 4);


%% Main Loop
start = Result.iter;

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
  if options.gpu
    res = hcube.ica_calc_S(Model.A, Result.D, Result.S);
    if res ~= 1
      error ('Error during computation on the GPU')
    end
  else
    Result.S = pinv(Model.A)*Result.D;
    %Ai = zeros(size(Model.A));
    %hcube.ica_pinv(Model.A, Ai);
    %Result.S = Ai*Result.D;
    %hcube.ica_calc_S (Model.A, Result.D, Result.S);
  end
  calcTimes(cT, 2) = toc(tstart);
  
  tstart = tic;
  [Model, Result] = adaptPrior(Model, Result, fitPar, hcube, options);
  calcTimes(cT, 3) = toc(tstart);
  
  if (i == start || isUpdatePoint (i, dispPar.updateFreq, fitPar))
    Result = updateDisplay(Model, Result, fitPar, dispPar);
  end
  
  tstart = tic;
  epsilon = interpIter (i, fitPar.iterPts, fitPar.epsilon);
  if options.gpu
    Model.A = updateAonGPU (Model, Result, epsilon, hcube);
  else   
    [dA, A] = calcDeltaA (Result.S, Model);
    Model.A = updateAwithDeltaA (A, dA, epsilon);
  end
  
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

function [A] = updateAonGPU (Model, Result, epsilon, hcube)
A = Model.A;
S = Result.S;
mu = Model.prior.mu;
beta  = Model.prior.beta;
sigma = Model.prior.sigma;
res = hcube.ica_update_A (A, S, mu, beta, sigma, epsilon);

if res ~= 1
  error ('Error during computation on the GPU')
end

end
