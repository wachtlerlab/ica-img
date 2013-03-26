function [m, n] = plotCreateGrid(M, landscape)

if floor(sqrt(M))^2 ~= M
  m = ceil(sqrt(M/2));
  n = ceil(M/m);
else
  m = sqrt(M);
  n = m;
end

if exist('landscape','var') && landscape
    c = m;
    m = n;
    n = c;
end

end

