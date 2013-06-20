function [ table ] = calcKLDTable( Acts )

L = length(Acts);
table = cell(L, 5);

for n = 1:L
    act = Acts{n};
    A = AfromActivations(act);
    table{n, 1} = A.name;
    table{n, 2} = calcAklDiv(A);
    table{n, 3} = mean(A.beta);
    table{n, 4} = mean(A.nkurt);
    table{n, 5} = mean(A.pkurt);
end

end

