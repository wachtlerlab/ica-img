% loginterp.m ----------------------------------------------------------
function y = loginterp(n,startN,stopN,ymin,ymax)
% lininterp -- Do interpolation of y between ymax and ymin on log scale.
%   Usage
%     x = lininterp(n,N,ymin,ymax)
%   Inputs
%     n        current iteration, must be between startN and stopN
%     startN   iteration to start iterpolation from
%     stopN    maximum iteration
%     ymin
%     ymax
%   Outputs
%     y     the iterpolated value between ymin and ymax (log scale)

if (stopN - startN < 1) 
  fprintf('\nloginterp: Invalid start/stop range [%d, %d]\n', startN, stopN);
end

if (n < startN | n > stopN)
  fprintf('\nloginterp: n=%d is out of the range [%d, %d]\n', ...
      n, startN, stopN);
  n = max(n, startN);
  n = min(n, stopN);
end

x = (n - startN)/(stopN - startN);
logy = (1-x)*log(ymin) + x*log(ymax);
y = exp(logy);

