function [ output_args ] = plotAbf( A )

if isa (A, 'struct')
    A = A.A;
end

[L,M] = size(A);
for m = 1:M
  nA(m) = norm(A(:,m));
end

[snA pidx] = sort(-nA);

figure()
for i=1:M
    idx = pidx(i);
    
    R = reshape (A(:,idx), 3, 7, 7);
    B = permute (R, [3 2 1]);
    
    k = abs(min(B(:)));
    p = max(B(:));    

    m = 0.99/(p+k);
    t = k * m;
    nim = m .* B + t;
    
    subplot(15,10,i);
    
    image(nim);
    axis off;
end

end

