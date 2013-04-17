function [ sortedA, idx, nA ] = sortAbf (A)

if isa (A, 'struct')
    A = A.A;
end

[~,M] = size(A);

nA = ones (1, M);

for m = 1:M
  nA(m) = norm(A(:,m));
end

[~, idx] = sort(-nA);
sortedA = A (:,idx);

end

