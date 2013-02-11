function [m, n] = plotCreateGrid(M)

if floor(sqrt(M))^2 ~= M
  m = ceil(sqrt(M/2));
  n = ceil(M/m);
else
  m = sqrt(M);
  n = m;
end

end

