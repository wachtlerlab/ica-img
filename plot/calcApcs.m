function [ pcs, of ] = calcApcs( A )

nbf = A.nbf;
pcs = zeros(2, nbf);
of = zeros(2, nbf);

for n=1:nbf
    
    dkl = A.dkl(:, :, n);
    x = dkl(:, 1);
    y = dkl(:, 2);
    
    data = [x y];
    data = data - repmat(mean(data), size(data, 1), 1);
    
    [pc,~,latent,~] = princomp(data);
    pc = pc(:,1) * latent (1);
    pcs(:, n) = pc;
    of(:, n) = [latent(1) latent(2)];

end

end

