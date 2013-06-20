function [ found ] = findAachromatic(A)

dkl = A.dkl;
[~, M] = size(A.rgb);

found = zeros(M, 1);
for idx=1:M
    x = dkl(:, 1, idx);
    y = dkl(:, 2, idx);
    
    [~, ix] = max(abs(x));
    [~, iy] = max(abs(y));
    
    %D = [sign(x(ix)), sign(y(iy))]
    D = [sign(x), sign(y)];
    U = unique(D, 'rows');
    
    
    nq = size(U, 1);
    if nq == 1
        found(idx) = 1;
    elseif nq == 2 && sum(sum(U,1) ~= [0 0])
       found(idx) = 2;
    end
end


end

