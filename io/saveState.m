function saveState( Model, fullResult, saveFreq )

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

nick = Model.id(1:7);
cfgid = Model.cfg(1:7);

stateDir = fullfile ('..', 'state', nick);
if exist (stateDir, 'dir') ~= 7
  mkdir (stateDir);
end

stateFileName = sprintf ('%s-%s-i%d', cfgid, nick, Result.iter);
stateFile = fullfile (stateDir, stateFileName);

fprintf('%5d: Saving state to %s\n', Result.iter, stateFile);
eval(['save ',stateFile,' Model Result']);

% save state with minimum number of bits/pixel in separate file
if ( Result.bits(Result.plotIter) == min(Result.bits(1:Result.plotIter)) )
  
 optFileName = sprintf('%s-%s-minbits', cfgid, nick);
 optFile = fullfile (stateDir, optFileName);
 eval(['save ',optFile,' Model Result']);
end


if (Result.iter == 1)
  return;

else

  % delete the previous state file.
  prevIter = max( 1, Result.iter - saveFreq );
  if (rem(prevIter, 10000) ~= 0)
    prevStateFileName = sprintf ('%s-%s-i%d.mat', cfgid, nick, prevIter);
    prevStateFile = fullfile (stateDir, prevStateFileName);
   
   if exist( prevStateFile, 'file' ) ~= 0
     delete( prevStateFile );
   end
  end
end
