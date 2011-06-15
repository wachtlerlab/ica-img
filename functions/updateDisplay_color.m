function Result = updateDisplay_color(Model, Result, fitPar, dispPar, initflag)

% display progress and update plots
% also do calculations for progress reports

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

persistent bfh

skipDC=1;

[L,M] = size(Model.A);
[L,N] = size(Result.D);

if Result.iter == 1
  Result.plotIter = 0;
  Result.updateIters = zeros(1);
  Result.logL        = zeros(1);
  Result.bits        = zeros(1);
  Result.avgSD       = zeros(1,M);
end

if (isempty(bfh) | exist('initflag','var') == 1)		% first call
  if dispPar.plotflag
%    for i=1:3
%      tilefig(i); clf;
%    end
%    figure(1);
%    bfh = plotBFs(Model.A,dispPar.maxPlotVecs,'zeroc','l2ord');
  end
end

Result.plotIter = Result.plotIter + 1;
Result.updateIters(Result.plotIter) = Result.iter;

logL = calcLogPXA(Result.D, Result.S, Model);

Result.logL(Result.plotIter) = logL / N;
Result.bits(Result.plotIter) = estBits(logL, 8, Result.D) / (L*N);

if dispPar.plotflag
%  figure(1);
%  bfh = plotBFs(Model.A,dispPar.maxPlotVecs,'zeroc','l2ord');

  set (0, 'CurrentFigure', 2);
  subplot(221), plot(Result.updateIters, Result.bits, '+');

  if skipDC
    subplot(222), pidx = plotAnorm(Model.A(:,2:M));
    pidx = [1, pidx+1];
  else
    subplot(222), pidx = plotAnorm(Model.A);
  end
  subplot(223), bar(Model.prior.beta(pidx));
  axis tight;
  ylabel('\beta');
  Result.avgSD = 0.5*std(Result.S') + 0.5*Result.avgSD;
  subplot(224), plotSvar(Result.S,pidx,Result.avgSD);

  set (0, 'CurrentFigure', 3);
  nbins = N/10;
  plotShist2(Result.S, nbins, [3, 4]);

  drawnow;
end

fprintf('\r%5d: logL = %4.2f  (%4.2f bits/pixel)\n', ...
    Result.iter, Result.logL(Result.plotIter), Result.bits(Result.plotIter) );

end