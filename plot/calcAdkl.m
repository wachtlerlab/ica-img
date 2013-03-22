function [ dkl ] = calcAdkl(A)

nbf   = A.nbf;
ps    = A.ps;
nchan = 3;

dkl = zeros(ps*ps, nchan, nbf);

for n=1:nbf
    rgb = A.rgb(:, n);
    
    slice = permute(reshape(rgb, nchan, ps*ps), [2 1]);
    L =  slice(:, 1);
    M =  slice(:, 2);
    S =  slice(:, 3);

    x = L-M;
    y = S-((L+M)/2);
    z = L+M;
    
    dkl(:, :, n) = [x, y, z];
end

end

