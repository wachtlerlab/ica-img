function [A, dts] = find_directions (X, N, eps)
X = repmat (X, 1, 100);
[m, n] = size (X);
X = X(:, randperm(n));

A = normBasis (rand (m, N));

if nargin < 3
  eps = 0.0001;
end

dts = zeros (m, N);
for k = 1:n
  S = normBasis (X(:, k));
  Sm = repmat (S, 1, N);
  d = dot (Sm, A);
  dts(k, :) = d;
  [~, idx] = max (abs(d));
  A(:,idx)=A(:,idx) + eps*(A(:,idx)-S);
  A = normBasis (A);
end

end

function [C] = normBasis (A)
C = cell2mat (cellfun (@(x) x/norm(x), num2cell (A,1), 'UniformOutput', false));
end