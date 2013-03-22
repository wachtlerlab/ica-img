function [ pcs ] = calcApcs( A )

nbf = A.nbf;
pcs = zeros(2, nbf);

for n=1:nbf
    
    dkl = A.dkl(:, :, n);
    x = dkl(:, 1);
    y = dkl(:, 2);
    
    [pc,~,latent,~] = princomp([x y]);
    pc = pc(:,1) * latent (1);
    pcs(:, n) = pc;

end

end

