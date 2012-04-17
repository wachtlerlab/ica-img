function varargout = getinterp(iter, iterpts, varargin)
% getinterp -- get interpolated points at iteration
%   Usage
%     [point1, point2, ...]  = getinterp(iter, iterpts, set1, set2, ...)
%   Inputs
%      iter            The current iteration.
%      iterpts         The set of iteration anchor points.
%      set1,set2,...   Variables to be interpolation   
%   Outputs
%      point1,poitn2   Interpolated points at current iteration.
  
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

if (iter < max(iterpts)) 
  a = max(find(iter >= iterpts));
  b = min(find(iter < iterpts));
  for i=1:nargin-2
    set = varargin{i};
    varargout(i) = {loginterp(iter, iterpts(a), iterpts(b), set(a), set(b))};
  end
else
  a = size(iterpts,2);
  for i=1:nargin-2
    set = varargin{i};
    varargout(i) = {set(a)};
  end
end
