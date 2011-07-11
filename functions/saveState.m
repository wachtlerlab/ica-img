function saveState( Model, fullResult, fitPar )

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

% Don't save data and coeffs to keep file size small.
% D is a random sample, so we can just resample.
Result = fullResult;
Result.X = [];
Result.D = [];
Result.S = [];
Result.priorS = [];

stateFile = sprintf('state/%s-state-i%d', Model.name, Result.iter);
fprintf('%5d: Saving state to %s\n', Result.iter, stateFile);
eval(['save ',stateFile,' Model Result']);

% save state with minimum number of bits/pixel in separate file
if ( Result.bits(Result.plotIter) == min(Result.bits(1:Result.plotIter)) )
 optFile = sprintf('state/%s-state-minbits', Model.name);
 eval(['save ',optFile,' Model Result']);
end


if (Result.iter == 1)
  return;

else

  % delete the previous state file.
  prevIter = max( 1, Result.iter - fitPar.saveFreq );
  if (rem(prevIter, 10000) ~= 0)
   prevStateFile = sprintf( 'state/%s-state-i%d.mat', Model.name, prevIter );
   if exist( prevStateFile, 'file' ) ~= 0
     delete( prevStateFile );
   end
  end
end
