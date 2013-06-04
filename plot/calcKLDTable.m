function [ table ] = calcKLDTable( Acts )

L = length(Acts);
table = cell(L, 2);
for n = 1:L
    act = Acts(n);
    A = AfromActivations(act);
    table{n, 1} = A.name;
    table{n, 2} = calcAklDiv(A);
end

end

