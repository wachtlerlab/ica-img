function Model = rescaleBfs(Model,Result)
% rescale lengths of basis functions for initial fit

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

sigma = std(Result.D(:));
fprintf('%5d: Rescaling basis functions.\n',Result.iter);
[L,M] = size(Model.A);
for m=1:M
  Model.A(:,m) = Model.A(:,m) * sigma;
end
