function [A] = AfromActivations(Act)

Model = Act.Model;
nchan = 3;


A = prepareAfromModel(Model, 3);
nbf = A.nbf;
ps  = A.ps;

meanact = mean(Act.mact, 3);

Ainit3 = zeros(ps*ps*nchan, nbf);
A.sml  = Ainit3;
A.lms  = Ainit3;
A.rgb  = Ainit3;

for n=1:nbf
    ompatch = meanact(:, n);
    sml = reshape(ompatch, nchan, ps, ps);
    lms = flipdim(reshape(ompatch, nchan, ps, ps), 1);
    rgb = 0.5 + 0.5 * (lms/max(abs(lms(:))));
    
    A.sml(:, n) = sml(:);
    A.lms(:, n) = lms(:);
    A.rgb(:, n) = rgb(:);
end

A.dkl = calcAdkl(A);
A.pcs = calcApcs(A);

end

%  ompatch = mean(opatchw, 2);
%     sml = reshape(flipdim(reshape(ompatch, 3, ps, ps), 1), bflen, 1);
%     rgb = 0.5 + 0.5 * (sml/max(abs(sml)));
% 
