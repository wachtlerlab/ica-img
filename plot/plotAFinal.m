function [ output_args ] = plotAFinal(A)

nbf = A.nbf;
pcs = zeros(2, nbf);

for n=1:nbf
    
    dkl = A.dkl(:, :, n);
    x = dkl(:, 1);
    y = dkl(:, 2);
    
    [rho, idx] = max(hypot(x, y));
    theta = atan2(y(idx), x(idx));
    pcs(:, n) = [theta, rho];
    
end

tt = pcs(1, :);
rr = pcs(2, :);

[hf_dir, hf_plane] = plotDirs([''], tt, rr, 0, 0);


end

